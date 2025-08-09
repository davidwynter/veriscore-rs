use clap::{Parser, Subcommand};
use anyhow::Result;
use veriscore_rs::{llm::openai::LlmClient, serper::Serper};
use veriscore_rs::server::{run_server, Engine};

#[derive(Parser)]
#[command(name="veriscore", version)]
struct Cli {
  #[command(subcommand)]
  cmd: Cmd,
  /// Where to read/write JSONL
  #[arg(long, default_value="./data")] data_dir: String,
  /// Cache DB path
  #[arg(long, default_value="./data/cache.sqlite")] cache_db: String,
}

#[derive(Subcommand)]
enum Cmd {
  /// End-to-end: extract -> retrieve -> verify
  Run { #[arg(long)] input_file: String, #[arg(long, default_value="llama-3.3-70b-instruct")] extract_model: String, #[arg(long, default_value="llama-3.3-70b-instruct")] verify_model: String, #[arg(long, default_value_t=64)] llm_concurrency: usize, #[arg(long, default_value_t=16)] search_concurrency: usize, },
  Extract { #[arg(long)] input_file: String, #[arg(long, default_value="llama-3.3-70b-instruct")] model: String, #[arg(long, default_value_t=64)] llm_concurrency: usize },
  Retrieve { #[arg(long)] input_file: String, #[arg(long, default_value_t=16)] search_concurrency: usize, #[arg(long, default_value_t=10)] search_res_num: usize },
  Verify { #[arg(long)] input_file: String, #[arg(long, default_value="llama-3.3-70b-instruct")] model: String, #[arg(long, default_value_t=64)] llm_concurrency: usize, #[arg(long, default_value_t=1)] label_n: u8 },
  Score   { #[arg(long)] input_file: String, #[arg(long)] k_median: usize, #[arg(long, default_value="skip")] abstentions: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let base = std::env::var("OPENAI_BASE_URL").ok();
    let key  = std::env::var("OPENAI_API_KEY").ok();
    let extract_model = std::env::var("EXTRACT_MODEL").unwrap_or("llama-3.3-70b-instruct".into());
    let verify_model  = std::env::var("VERIFY_MODEL").unwrap_or("llama-3.3-70b-instruct".into());

    let llm_extract = LlmClient::new(extract_model, base.clone(), key.clone(), /*max_concurrency*/128);
    let llm_verify  = LlmClient::new(verify_model,  base,          key,         /*max_concurrency*/128);

    let serper_key = std::env::var("SERPER_API_KEY").expect("SERPER_API_KEY required");
    let serper = Serper::new(serper_key, /*qps*/20, /*top_k*/8, /*timeout_ms*/3500);

    let engine = Engine {
        llm_extract, llm_verify, serper,
        search_concurrency: 64,
        llm_concurrency: 128,
    };

    run_server(engine, "0.0.0.0:8088").await
}
