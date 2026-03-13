use anyhow::Result;
use std::sync::Arc;
use veriscore_runtime::pipeline::StatelessPipeline;

use crate::reward_types::{RewardDetail, RewardRequest, RewardResponse};

#[derive(Clone)]
pub struct RewardEngine {
    pipeline: Arc<StatelessPipeline>,
}

impl RewardEngine {
    pub fn new(pipeline: Arc<StatelessPipeline>) -> Self {
        Self { pipeline }
    }

    pub async fn score_batch(&self, request: RewardRequest, include_details: bool) -> Result<RewardResponse> {
        let mut rewards = Vec::with_capacity(request.completions.len());
        let mut details = Vec::new();

        for record in request.completions.iter() {
            let (_verification, score) = self.pipeline
                .verify_and_score(record, request.binary, request.k_median)
                .await?;
            rewards.push(score.f1);
            if include_details {
                details.push(RewardDetail {
                    supported: score.supported,
                    total: score.total,
                    precision: score.precision,
                    recall: score.recall,
                    f1: score.f1,
                });
            }
        }

        Ok(RewardResponse {
            rewards,
            details: include_details.then_some(details),
        })
    }
}
