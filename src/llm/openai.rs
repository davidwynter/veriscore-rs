use anyhow::Result;
use async_openai::{config::OpenAIConfig, Client, types::{ChatCompletionRequestMessage, CreateChatCompletionRequestArgs}};
use futures::{stream, StreamExt};
use super::Llm;
use anyhow::Result;
use async_openai::types::ChatCompletionRequestMessage;

#[async_trait::async_trait]
impl Llm for LlmClient {
    async fn chat_many(&self, prompts: Vec<Vec<ChatCompletionRequestMessage>>) -> Result<Vec<String>> {
        // (see 0.2 below for the improved stable-order implementation)
        self.chat_many(prompts).await
    }
}

#[derive(Clone)]
pub struct LlmClient {
    client: Client<OpenAIConfig>,
    model: String,
    max_concurrency: usize,
}

impl LlmClient {
    pub fn new(model: String, base_url: Option<String>, api_key: Option<String>, max_concurrency: usize) -> Self {
        let mut cfg = OpenAIConfig::default();
        if let Some(url) = base_url { cfg = cfg.with_api_base(url); }
        if let Some(key) = api_key { cfg = cfg.with_api_key(key); }
        let client = Client::with_config(cfg);
        Self { client, model, max_concurrency }
    }

    pub async fn chat_many(&self, prompts: Vec<Vec<ChatCompletionRequestMessage>>) -> Result<Vec<String>> {
        use futures::{stream, StreamExt};
    
        let reqs = prompts.into_iter().enumerate().map(|(idx, messages)| {
            let client = self.client.clone();
            let model = self.model.clone();
            async move {
                let req = CreateChatCompletionRequestArgs::default()
                    .model(model)
                    .messages(messages)
                    .build()
                    .unwrap();
                let resp = client.chat().create(req).await?;
                let text = resp.choices[0].message.content.clone().unwrap_or_default();
                Ok::<_, anyhow::Error>((idx, text))
            }
        });
    
        let mut out = stream::iter(reqs)
            .buffer_unordered(self.max_concurrency)
            .collect::<Vec<_>>()
            .await;
    
        out.sort_by_key(|r| r.as_ref().map(|(i, _)| *i).unwrap_or(usize::MAX));
        let mut texts = Vec::with_capacity(out.len());
        for r in out {
            let (_, t) = r?;
            texts.push(t);
        }
        Ok(texts)
    }
}
