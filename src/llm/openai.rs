// src/llm/openai.rs
use anyhow::Result;
use async_openai::{config::OpenAIConfig, Client, types::{ChatCompletionRequestMessage, CreateChatCompletionRequestArgs}};
use futures::{stream, StreamExt};

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
        // Generic concurrent fan-out (portable across OpenAI, Azure, Matrix/vLLM)
        let reqs = prompts.into_iter().map(|messages| {
            let client = self.client.clone();
            let model = self.model.clone();
            async move {
                let req = CreateChatCompletionRequestArgs::default()
                    .model(model)
                    .messages(messages)
                    .build()
                    .unwrap();
                let resp = client.chat().create(req).await?;
                Ok::<_, anyhow::Error>(resp.choices[0].message.content.clone().unwrap_or_default())
            }
        });

        let results = stream::iter(reqs)
            .buffer_unordered(self.max_concurrency)  // micro-batching comes from server-side continuous batching
            .collect::<Vec<_>>()
            .await;
        // Unwrap preserving order (best effort)
        let mut out = Vec::with_capacity(results.len());
        for r in results { out.push(r?); }
        Ok(out)
    }
}
