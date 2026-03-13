use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputRecord {
    pub question: Option<String>,       // optional; if present, QA mode
    pub response: String,               // model output to evaluate
    pub model: Option<String>,          // generator id
    pub prompt_source: Option<String>,  // dataset tag/domain
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedClaimsRecord {
    #[serde(flatten)]
    pub input: InputRecord,
    pub prompt_tok_cnt: Option<u32>,
    pub response_tok_cnt: Option<u32>,
    pub abstained: bool,                // when extractor refuses
    pub claim_list: Vec<Vec<String>>,   // claims per-snippet (sliding window)
    pub all_claims: Vec<String>,        // flattened
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem { pub title: String, pub snippet: String, pub link: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRecord {
    #[serde(flatten)]
    pub claims: ExtractedClaimsRecord,
    pub claim_snippets_dict: Vec<(String, Vec<EvidenceItem>)>, // (claim, evidence[])
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationLabel { Supported, Unsupported /* optionally: Contradicted, Inconclusive */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimVerification {
    pub claim: String,
    pub search_results: Vec<EvidenceItem>,   // concatenated or per-item
    pub verification_result: VerificationLabel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRecord {
    #[serde(flatten)]
    pub evidence: EvidenceRecord,
    pub claim_verification_result: Vec<ClaimVerification>,
}
