use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub llm_concurrency: usize,
    pub evidence_concurrency: usize,
    pub evidence_top_k: usize,
    pub verification_binary: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            llm_concurrency: 64,
            evidence_concurrency: 32,
            evidence_top_k: 8,
            verification_binary: true,
        }
    }
}
