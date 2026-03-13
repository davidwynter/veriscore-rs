use crate::segment::{segment_sentences, sliding_windows};
use crate::types::*;
use anyhow::Result;
use async_openai::types::ChatCompletionRequestMessage;
use veriscore_llm::traits::Llm;

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InputRecord;
    use async_openai::types::ChatCompletionRequestMessage;

    struct FakeLlm {
        outputs: Vec<String>,
    }

    #[async_trait::async_trait]
    impl veriscore_llm::traits::Llm for FakeLlm {
        async fn chat_many(
            &self,
            _prompts: Vec<Vec<ChatCompletionRequestMessage>>,
        ) -> anyhow::Result<Vec<String>> {
            Ok(self.outputs.clone())
        }
    }

    #[tokio::test]
    async fn extract_record_flattens_claims() {
        let llm = FakeLlm {
            outputs: vec![
                r#"["Claim A1","Claim A2"]"#.to_string(),
                r#"[]"#.to_string(),
                r#"["Claim C1"]"#.to_string(),
            ],
        };

        let rec = InputRecord {
            question: None,
            response: "Sentence A. Sentence B. Sentence C.".to_string(),
            model: None,
            prompt_source: None,
        };

        let out = extract_record(&llm, &rec).await.unwrap();

        assert_eq!(out.claim_list.len(), 3);
        assert_eq!(
            out.all_claims,
            vec![
                "Claim A1".to_string(),
                "Claim A2".to_string(),
                "Claim C1".to_string()
            ]
        );
        assert!(!out.abstained);
    }

    #[tokio::test]
    async fn extract_record_handles_invalid_json_as_empty() {
        let llm = FakeLlm {
            outputs: vec![
                "not-json".to_string(),
                r#"["Claim B1"]"#.to_string(),
            ],
        };

        let rec = InputRecord {
            question: None,
            response: "Sentence A. Sentence B.".to_string(),
            model: None,
            prompt_source: None,
        };

        let out = extract_record(&llm, &rec).await.unwrap();

        assert_eq!(out.claim_list.len(), 2);
        assert_eq!(out.claim_list[0], Vec::<String>::new());
        assert_eq!(out.claim_list[1], vec!["Claim B1".to_string()]);
        assert_eq!(out.all_claims, vec!["Claim B1".to_string()]);
    }

    #[tokio::test]
    async fn extract_record_supports_qa_mode() {
        let llm = FakeLlm {
            outputs: vec![
                r#"["Answer claim 1"]"#.to_string(),
            ],
        };

        let rec = InputRecord {
            question: Some("What is the capital of France?".to_string()),
            response: "Paris is the capital of France.".to_string(),
            model: None,
            prompt_source: None,
        };

        let out = extract_record(&llm, &rec).await.unwrap();

        assert_eq!(out.claim_list.len(), 1);
        assert_eq!(out.all_claims, vec!["Answer claim 1".to_string()]);
    }
}