use serde::{Deserialize, Serialize};
use veriscore_core::types::InputRecord;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RewardRequest {
    pub group_id: String,
    pub k_median: usize,
    #[serde(default = "default_binary")]
    pub binary: bool,
    pub completions: Vec<InputRecord>,
}

fn default_binary() -> bool { true }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RewardDetail {
    pub supported: usize,
    pub total: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RewardResponse {
    pub rewards: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Vec<RewardDetail>>,
}
