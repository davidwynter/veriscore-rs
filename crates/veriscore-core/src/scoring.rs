use crate::types::{VerificationLabel, VerificationRecord};

pub struct ScoreConf { pub k: usize, pub abstentions_zero: bool }

pub struct PerResponseScore { pub supported: usize, pub total: usize, pub precision: f32, pub recall: f32, pub f1: f32 }

pub fn score_response(vr: &VerificationRecord, k: usize) -> PerResponseScore {
    let supported = vr.claim_verification_result.iter().filter(|c| matches!(c.verification_result, VerificationLabel::Supported)).count();
    let total = vr.claim_verification_result.len().max(1);
    let precision = supported as f32 / total as f32;
    // recall uses K as the target count for perfect recall
    let recall = (supported as f32 / k as f32).min(1.0);
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    PerResponseScore { supported, total, precision, recall, f1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ClaimVerification, EvidenceItem, EvidenceRecord, ExtractedClaimsRecord, InputRecord,
        VerificationLabel, VerificationRecord,
    };

    fn mk_record(supported: usize, total: usize) -> VerificationRecord {
        let claim_verification_result = (0..total)
            .map(|i| ClaimVerification {
                claim: format!("claim-{i}"),
                search_results: vec![EvidenceItem {
                    title: "title".to_string(),
                    snippet: "snippet".to_string(),
                    link: "link".to_string(),
                }],
                verification_result: if i < supported {
                    VerificationLabel::Supported
                } else {
                    VerificationLabel::Unsupported
                },
            })
            .collect();

        VerificationRecord {
            evidence: EvidenceRecord {
                claims: ExtractedClaimsRecord {
                    input: InputRecord {
                        question: None,
                        response: "response".to_string(),
                        model: None,
                        prompt_source: None,
                    },
                    prompt_tok_cnt: None,
                    response_tok_cnt: None,
                    abstained: false,
                    claim_list: vec![],
                    all_claims: vec![],
                },
                claim_snippets_dict: vec![],
            },
            claim_verification_result,
        }
    }

    #[test]
    fn score_response_basic_case() {
        let vr = mk_record(3, 6);
        let s = score_response(&vr, 6);

        assert_eq!(s.supported, 3);
        assert_eq!(s.total, 6);
        assert!((s.precision - 0.5).abs() < 1e-6);
        assert!((s.recall - 0.5).abs() < 1e-6);
        assert!((s.f1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn score_response_caps_recall_at_one() {
        let vr = mk_record(10, 10);
        let s = score_response(&vr, 6);

        assert_eq!(s.supported, 10);
        assert!((s.recall - 1.0).abs() < 1e-6);
        assert!(s.f1 <= 1.0);
    }

    #[test]
    fn score_response_handles_zero_supported() {
        let vr = mk_record(0, 5);
        let s = score_response(&vr, 5);

        assert_eq!(s.supported, 0);
        assert_eq!(s.total, 5);
        assert!((s.precision - 0.0).abs() < 1e-6);
        assert!((s.recall - 0.0).abs() < 1e-6);
        assert!((s.f1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn score_response_handles_zero_claims_defensively() {
        let vr = mk_record(0, 0);
        let s = score_response(&vr, 5);

        assert_eq!(s.supported, 0);
        assert_eq!(s.total, 1);
        assert!((s.precision - 0.0).abs() < 1e-6);
    }
}