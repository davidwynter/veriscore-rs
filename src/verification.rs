use crate::{types::*, llm::openai::LlmClient};
use anyhow::Result;
use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs};

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

pub async fn verify_record(client: &LlmClient, ev: EvidenceRecord, binary: bool, concurrency: usize)
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
