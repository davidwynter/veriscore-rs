use crate::{types::*, llm::openai::LlmClient};
use anyhow::Result;
use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs};
use crate::llm::Llm;

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

pub async fn verify_record(client: &dyn Llm, ev: EvidenceRecord, binary: bool, concurrency: usize)
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
    use crate::types::*;
    use async_openai::types::ChatCompletionRequestMessage;

    struct FakeVerifier {
        pub labels: Vec<&'static str>,
    }
    #[async_trait::async_trait]
    impl crate::llm::Llm for FakeVerifier {
        async fn chat_many(&self, _prompts: Vec<Vec<ChatCompletionRequestMessage>>) -> anyhow::Result<Vec<String>> {
            Ok(self.labels.iter().map(|l| format!(r#"{{"label":"{}"}}"#, l)).collect())
        }
    }

    #[tokio::test]
    async fn verify_collapses_to_binary() {
        let rec = EvidenceRecord {
            claims: ExtractedClaimsRecord {
                input: InputRecord { question: None, response: "r".into(), model: None, prompt_source: None },
                prompt_tok_cnt: None, response_tok_cnt: None, abstained: false,
                claim_list: vec![], all_claims: vec!["c1".into(),"c2".into(),"c3".into()],
            },
            claim_snippets_dict: vec![
                ("c1".into(), vec![EvidenceItem{title:"t".into(), snippet:"s".into(), link:"l".into()}]),
                ("c2".into(), vec![EvidenceItem{title:"t".into(), snippet:"s".into(), link:"l".into()}]),
                ("c3".into(), vec![EvidenceItem{title:"t".into(), snippet:"s".into(), link:"l".into()}]),
            ],
        };
        let fake = FakeVerifier { labels: vec!["supported","contradicted","inconclusive"] };
        let out = verify_record(&fake, rec, /*binary=*/true, 16).await.unwrap();
        let labs: Vec<_> = out.claim_verification_result.iter().map(|c| &c.verification_result).collect();
        assert!(matches!(labs[0], VerificationLabel::Supported));
        assert!(matches!(labs[1], VerificationLabel::Unsupported));
        assert!(matches!(labs[2], VerificationLabel::Unsupported));
    }
}
