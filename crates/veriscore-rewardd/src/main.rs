use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;
use veriscore_llm::{BatchedLlm, MicroBatchConfig, OpenAiCompatibleLlm};
use veriscore_llm::cache::LlmCache;
use veriscore_reward::{build_router, RewardEngine};
use veriscore_reward::reward_api::RewardApiState;
use veriscore_runtime::pipeline::StatelessPipeline;
use veriscore_web::cache::WebCache;
use veriscore_web::serper::Serper;
use veriscore_web::WebEvidenceProvider;

#[derive(Debug, Parser)]
#[command(name = "veriscore-rewardd")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8088")]
    listen: String,

    #[arg(long, env = "OPENAI_BASE_URL")]
    openai_base_url: Option<String>,

    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,

    #[arg(long, env = "EXTRACT_MODEL", default_value = "llama-3.3-70b-instruct")]
    extract_model: String,

    #[arg(long, env = "VERIFY_MODEL", default_value = "llama-3.3-70b-instruct")]
    verify_model: String,

    #[arg(long, env = "SERPER_API_KEY")]
    serper_api_key: String,

    #[arg(long, default_value_t = 64)]
    llm_concurrency: usize,

    #[arg(long, default_value_t = 32)]
    search_concurrency: usize,

    #[arg(long, default_value_t = 8)]
    serper_top_k: usize,

    #[arg(long, default_value_t = 32)]
    max_batch_size: usize,

    #[arg(long, default_value_t = 10)]
    max_batch_wait_ms: u64,

    #[arg(long, default_value = "./llm_cache.sqlite")]
    llm_cache_db: String,

    #[arg(long, default_value = "./web_cache.sqlite")]
    web_cache_db: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let args = Args::parse();

    let llm_cache = Arc::new(LlmCache::open(&args.llm_cache_db)?);
    let web_cache = Arc::new(WebCache::open(&args.web_cache_db)?);

    let extract_raw = Arc::new(OpenAiCompatibleLlm::new(
        args.extract_model,
        args.openai_base_url.clone(),
        args.openai_api_key.clone(),
        args.llm_concurrency,
        Some(llm_cache.clone()),
    ));
    let verify_raw = Arc::new(OpenAiCompatibleLlm::new(
        args.verify_model,
        args.openai_base_url,
        args.openai_api_key,
        args.llm_concurrency,
        Some(llm_cache),
    ));

    let extract_llm = Arc::new(BatchedLlm::spawn(
        extract_raw,
        MicroBatchConfig {
            max_batch_size: args.max_batch_size,
            max_wait: std::time::Duration::from_millis(args.max_batch_wait_ms),
            queue_capacity: 4096,
        },
    ));
    let verify_llm = Arc::new(BatchedLlm::spawn(
        verify_raw,
        MicroBatchConfig {
            max_batch_size: args.max_batch_size,
            max_wait: std::time::Duration::from_millis(args.max_batch_wait_ms),
            queue_capacity: 4096,
        },
    ));

    let serper_http = reqwest::Client::new();
    let serper = Arc::new(Serper::new(serper_http, args.serper_api_key, args.serper_top_k));
    let evidence = Arc::new(WebEvidenceProvider::new(
        serper,
        args.serper_top_k,
        args.search_concurrency,
        Some(web_cache),
    ));

    let pipeline = Arc::new(StatelessPipeline {
        extractor: extract_llm,
        verifier: verify_llm,
        evidence,
    });
    let engine = Arc::new(RewardEngine::new(pipeline));
    let router = build_router(RewardApiState { engine }).layer(TraceLayer::new_for_http());

    let listener = TcpListener::bind(&args.listen).await?;
    tracing::info!(listen = %args.listen, "starting veriscore-rewardd");
    axum::serve(listener, router).await?;
    Ok(())
}
