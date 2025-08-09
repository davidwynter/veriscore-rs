// src/server.rs
use axum::{routing::post, Json, Router, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use anyhow::Result;

use crate::types::{InputRecord, VerificationRecord};
use crate::extraction::extract_record;
use crate::serper::Serper;
use crate::verification::verify_record;
use crate::scoring::score_response;
use crate::llm::openai::LlmClient;
use futures::{stream, StreamExt};

#[derive(Clone)]
pub struct Engine {
    llm_extract: LlmClient,
    llm_verify: LlmClient,
    serper: Serper,
    search_concurrency: usize,
    llm_concurrency: usize,
}

#[derive(Deserialize)]
struct RewardReq {
    group_id: String,
    k_median: usize,
    binary: bool,
    completions: Vec<InputRecord>,
}

#[derive(Serialize)]
struct RewardResp { rewards: Vec<f32> }

#[axum::debug_handler]
async fn reward_batch(
    State(engine): State<Arc<Engine>>,
    Json(req): Json<RewardReq>,
) -> Result<Json<RewardResp>, axum::http::StatusCode> {
    // For each completion: extract -> retrieve -> verify -> score
    let tasks = req.completions.iter().map(|inp| {
        let engine = engine.clone();
        async move {
            let extracted = extract_record(&engine.llm_extract, inp).await?;
            let evid = crate::retrieve::retrieve_for_record(&engine.serper, extracted, engine.search_concurrency).await?;
            let verified = verify_record(&engine.llm_verify, evid, req.binary, engine.llm_concurrency).await?;
            Ok::<VerificationRecord, anyhow::Error>(verified)
        }
    });

    let mut rewards = Vec::with_capacity(req.completions.len());
    let results = stream::iter(tasks).buffer_unordered(engine.llm_concurrency).collect::<Vec<_>>().await;

    for r in results {
        match r {
            Ok(vr) => {
                let s = score_response(&vr, req.k_median);
                rewards.push(s.f1); // or supported_fraction; choose your scalar here
            }
            Err(_) => rewards.push(0.0), // neutral fallback on failure/timeout
        }
    }
    Ok(Json(RewardResp { rewards }))
}

pub async fn run_server(engine: Engine, addr: &str) -> anyhow::Result<()> {
    let engine = Arc::new(engine);
    let app = Router::new()
        .route("/grpo/reward_batch", post(reward_batch))
        .with_state(engine);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
