use crate::{segment::{segment_sentences, sliding_windows, SlidingWinCfg}, types::*, llm::openai::LlmClient};
use anyhow::Result;
use async_openai::types::ChatCompletionRequestMessage;
use crate::llm::Llm;

fn build_extraction_prompt(win: &str) -> Vec<ChatCompletionRequestMessage> {
    use async_openai::types::{ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs};
    let system = ChatCompletionRequestSystemMessageArgs::default()
        .content("You are an expert at extracting verifiable factual claims. Extract only verifiable claims; ignore unverifiable content (opinions, advice, fiction). Return a JSON array of strings.")
        .build().unwrap().into();
    let user = ChatCompletionRequestUserMessageArgs::default()
        .content(format!("Text window:\n{win}\n\nReturn JSON array of verifiable claims."))
        .build().unwrap().into();
    vec![system, user]
}

pub async fn extract_record(client: &dyn Llm, rec: &InputRecord) -> Result<ExtractedClaimsRecord> {
    let sents = segment_sentences(&rec.response);
    let wins = sliding_windows(rec.question.as_deref(), &sents, crate::segment::SlidingWinCfg { left: 3, right: 1, qa_mode: rec.question.is_some() });

    let prompts = wins.iter().map(|w| build_extraction_prompt(w)).collect::<Vec<_>>();
    let raw = client.chat_many(prompts).await?;

    let mut claim_list = Vec::with_capacity(raw.len());
    let mut all_claims = Vec::new();
    for r in raw {
        let claims: Vec<String> = serde_json::from_str(r.trim()).unwrap_or_default();
        all_claims.extend(claims.clone());
        claim_list.push(claims);
    }

    Ok(ExtractedClaimsRecord {
        input: rec.clone(),
        prompt_tok_cnt: None, response_tok_cnt: None,
        abstained: false,
        claim_list, all_claims,
    })

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs, ChatCompletionRequestSystemMessageArgs};
    use crate::types::InputRecord;

    struct FakeLlmOnce;
    #[async_trait::async_trait]
    impl crate::llm::Llm for FakeLlmOnce {
        async fn chat_many(&self, _prompts: Vec<Vec<ChatCompletionRequestMessage>>) -> anyhow::Result<Vec<String>> {
            // Return a JSON array of claims for each window; simulate 3 windows
            Ok(vec![
                r#"["Claim A1","Claim A2"]"#.into(),
                r#"[]"#.into(),
                r#"["Claim C1"]"#.into(),
            ])
        }
    }

    #[tokio::test]
    async fn extract_flattens_and_groups() {
        let rec = InputRecord { question: None, response: "A. B. C.", model: None, prompt_source: None };
        let out = extract_record(&FakeLlmOnce, &rec).await.unwrap();
        assert_eq!(out.claim_list.len(), 3);
        assert_eq!(out.all_claims, vec!["Claim A1","Claim A2","Claim C1"]);
        assert!(!out.abstained);
    }
}
