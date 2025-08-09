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
