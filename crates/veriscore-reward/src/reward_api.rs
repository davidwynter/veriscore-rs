use std::sync::Arc;

use axum::{extract::{Query, State}, response::IntoResponse, routing::post, Json, Router};
use serde::Deserialize;

use crate::reward_engine::RewardEngine;
use crate::reward_types::{RewardRequest, RewardResponse};

#[derive(Debug, Deserialize)]
pub struct RewardApiQuery {
    #[serde(default)]
    pub include_details: bool,
}

#[derive(Clone)]
pub struct RewardApiState {
    pub engine: Arc<RewardEngine>,
}

pub fn build_router(state: RewardApiState) -> Router {
    Router::new()
        .route("/healthz", post(healthz).get(healthz))
        .route("/grpo/reward_batch", post(reward_batch))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({"ok": true}))
}

pub async fn reward_batch(
    State(state): State<RewardApiState>,
    Query(query): Query<RewardApiQuery>,
    Json(request): Json<RewardRequest>,
) -> Result<Json<RewardResponse>, (axum::http::StatusCode, String)> {
    state.engine
        .score_batch(request, query.include_details)
        .await
        .map(Json)
        .map_err(internal_error)
}

fn internal_error(err: anyhow::Error) -> (axum::http::StatusCode, String) {
    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
}
