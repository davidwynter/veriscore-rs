use anyhow::Result;
use std::sync::Arc;
use veriscore_core::extraction::extract_record;
use veriscore_core::scoring::{score_response, PerResponseScore};
use veriscore_core::types::{InputRecord, VerificationRecord};
use veriscore_core::verification::verify_record;
use veriscore_llm::traits::Llm;
use veriscore_web::web_evidence::EvidenceProvider;

pub struct StatelessPipeline {
    pub extractor: Arc<dyn Llm>,
    pub verifier: Arc<dyn Llm>,
    pub evidence: Arc<dyn EvidenceProvider>,
}

impl StatelessPipeline {
    pub async fn verify_and_score(
        &self,
        record: &InputRecord,
        binary: bool,
        k_median: usize,
    ) -> Result<(VerificationRecord, PerResponseScore)> {
        let extracted = extract_record(self.extractor.as_ref(), record).await?;
        let evidence_rows = self.evidence.fetch_evidence_for_claims(&extracted.all_claims).await?;
        let evidence_record = veriscore_core::types::EvidenceRecord {
            claims: extracted,
            claim_snippets_dict: evidence_rows,
        };
        let verification = verify_record(self.verifier.as_ref(), evidence_record, binary, 1).await?;
        let score = score_response(&verification, k_median);
        Ok((verification, score))
    }
}
