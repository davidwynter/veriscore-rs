pub mod types;
pub mod jsonl;
pub mod segment;
pub mod extraction;
pub mod verification;
pub mod scoring;
pub mod util;

pub use types::{
    ClaimVerification,
    EvidenceItem,
    EvidenceRecord,
    ExtractedClaimsRecord,
    InputRecord,
    VerificationLabel,
    VerificationRecord,
};

pub use extraction::extract_record;
pub use verification::verify_record;
pub use scoring::{score_response, PerResponseScore, ScoreConf};

use anyhow::Result;

/// Evidence provider abstraction for the OSS core.
///
/// The open-source workspace ships the `WebEvidenceProvider` implementation
/// in the `veriscore-web` crate. Private/commercial downstream products can
/// implement this trait for other evidence sources without changing core logic.
#[async_trait::async_trait]
pub trait EvidenceProvider: Send + Sync {
    async fn fetch_evidence_for_claims(
        &self,
        claims: &[String],
    ) -> Result<Vec<(String, Vec<EvidenceItem>)>>;
}