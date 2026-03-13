use crate::types::*;
use anyhow::Result;
use async_openai::types::{
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs,
};
use veriscore_llm::traits::Llm;

fn build_verify_prompt(claim: &str, hits: &[EvidenceItem], binary: bool) -> Vec<ChatCompletionRequestMessage> {
    let label_desc = if binary {
        "Return JSON: {\"label\": \"supported\" | \"unsupported\", \"rationale\": \"...\"}"
    } else {
        "Return JSON: {\"label\": \"supported\" | \"contradicted\" | \"inconclusive\", \"rationale\": \"...\"}"
    };
    let evidence = hits.iter().map(|h| format!("- {} [{}]\n{}", h.title, h.link, h.snippet)).collect::<Vec<_>>().join("\n");
    let sys = ChatCompletionRequestSystemMessageArgs::default()
        .content("You are a meticulous fact checker. Judge the claim ONLY using the provided web snippets.")
        .build().unwrap().into();
    let usr = ChatCompletionRequestUserMessageArgs::default()
        .content(format!("Claim:\n{claim}\n\nEvidence:\n{evidence}\n\n{label_desc}"))
        .build().unwrap().into();
    vec![sys, usr]
}

pub async fn verify_record(client: &dyn Llm, ev: EvidenceRecord, binary: bool, _concurrency: usize)
-> Result<VerificationRecord> {
    let prompts = ev.claim_snippets_dict.iter().map(|(c, hits)| build_verify_prompt(c, hits, binary)).collect::<Vec<_>>();
    let outs = client.chat_many(prompts).await?;
    let mut results = Vec::with_capacity(outs.len());

    for (i, out) in outs.into_iter().enumerate() {
        let obj: serde_json::Value = serde_json::from_str(&out).unwrap_or_else(|_| serde_json::json!({"label":"inconclusive"}));
        let label = obj.get("label").and_then(|v| v.as_str()).unwrap_or("inconclusive");
        let mapped = match label {
            "supported" => VerificationLabel::Supported,
            _ => VerificationLabel::Unsupported, // ternary collapsed to binary per paper
        };
        let (claim, hits) = &ev.claim_snippets_dict[i];
        results.push(ClaimVerification { claim: claim.clone(), search_results: hits.clone(), verification_result: mapped });
    }

    Ok(VerificationRecord { evidence: ev, claim_verification_result: results })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        EvidenceItem, EvidenceRecord, ExtractedClaimsRecord, InputRecord,
        VerificationLabel,
    };
    use async_openai::types::ChatCompletionRequestMessage;

    struct FakeVerifier {
        outputs: Vec<String>,
    }

    #[async_trait::async_trait]
    impl veriscore_llm::traits::Llm for FakeVerifier {
        async fn chat_many(
            &self,
            _prompts: Vec<Vec<ChatCompletionRequestMessage>>,
        ) -> anyhow::Result<Vec<String>> {
            Ok(self.outputs.clone())
        }
    }

    fn mk_evidence_record() -> EvidenceRecord {
        EvidenceRecord {
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
                all_claims: vec![
                    "Claim 1".to_string(),
                    "Claim 2".to_string(),
                    "Claim 3".to_string(),
                ],
            },
            claim_snippets_dict: vec![
                (
                    "Claim 1".to_string(),
                    vec![EvidenceItem {
                        title: "t1".to_string(),
                        snippet: "s1".to_string(),
                        link: "l1".to_string(),
                    }],
                ),
                (
                    "Claim 2".to_string(),
                    vec![EvidenceItem {
                        title: "t2".to_string(),
                        snippet: "s2".to_string(),
                        link: "l2".to_string(),
                    }],
                ),
                (
                    "Claim 3".to_string(),
                    vec![EvidenceItem {
                        title: "t3".to_string(),
                        snippet: "s3".to_string(),
                        link: "l3".to_string(),
                    }],
                ),
            ],
        }
    }

    #[tokio::test]
    async fn verify_record_binary_mode_collapses_non_supported() {
        let llm = FakeVerifier {
            outputs: vec![
                r#"{"label":"supported"}"#.to_string(),
                r#"{"label":"contradicted"}"#.to_string(),
                r#"{"label":"inconclusive"}"#.to_string(),
            ],
        };

        let ev = mk_evidence_record();
        let out = verify_record(&llm, ev, true, 8).await.unwrap();

        assert_eq!(out.claim_verification_result.len(), 3);
        assert!(matches!(
            out.claim_verification_result[0].verification_result,
            VerificationLabel::Supported
        ));
        assert!(matches!(
            out.claim_verification_result[1].verification_result,
            VerificationLabel::Unsupported
        ));
        assert!(matches!(
            out.claim_verification_result[2].verification_result,
            VerificationLabel::Unsupported
        ));
    }

    #[tokio::test]
    async fn verify_record_defaults_invalid_json_to_unsupported() {
        let llm = FakeVerifier {
            outputs: vec![
                "bad-json".to_string(),
                r#"{"label":"supported"}"#.to_string(),
                r#"{"label":"unsupported"}"#.to_string(),
            ],
        };

        let ev = mk_evidence_record();
        let out = verify_record(&llm, ev, true, 8).await.unwrap();

        assert!(matches!(
            out.claim_verification_result[0].verification_result,
            VerificationLabel::Unsupported
        ));
        assert!(matches!(
            out.claim_verification_result[1].verification_result,
            VerificationLabel::Supported
        ));
        assert!(matches!(
            out.claim_verification_result[2].verification_result,
            VerificationLabel::Unsupported
        ));
    }

    #[tokio::test]
    async fn verify_record_preserves_claim_text_and_evidence() {
        let llm = FakeVerifier {
            outputs: vec![
                r#"{"label":"supported"}"#.to_string(),
                r#"{"label":"supported"}"#.to_string(),
                r#"{"label":"supported"}"#.to_string(),
            ],
        };

        let ev = mk_evidence_record();
        let out = verify_record(&llm, ev, true, 8).await.unwrap();

        assert_eq!(out.claim_verification_result[0].claim, "Claim 1");
        assert_eq!(out.claim_verification_result[1].search_results.len(), 1);
        assert_eq!(out.claim_verification_result[2].search_results[0].title, "t3");
    }
}