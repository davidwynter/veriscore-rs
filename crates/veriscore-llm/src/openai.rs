use crate::cache::LlmCache;
use crate::traits::Llm;
use anyhow::{Context, Result};
use async_openai::config::OpenAIConfig;
use async_openai::types::{ChatCompletionRequestMessage, CreateChatCompletionRequestArgs};
use async_openai::Client;
use futures::{stream, StreamExt};
use std::sync::Arc;

#[derive(Clone)]
pub struct OpenAiCompatibleLlm {
    client: Client<OpenAIConfig>,
    model: String,
    max_concurrency: usize,
    cache: Option<Arc<LlmCache>>,
}

impl OpenAiCompatibleLlm {
    pub fn new(
        model: impl Into<String>,
        api_base: Option<String>,
        api_key: Option<String>,
        max_concurrency: usize,
        cache: Option<Arc<LlmCache>>,
    ) -> Self {
        let mut cfg = OpenAIConfig::default();
        if let Some(api_base) = api_base {
            cfg = cfg.with_api_base(api_base);
        }
        if let Some(api_key) = api_key {
            cfg = cfg.with_api_key(api_key);
        }
        Self {
            client: Client::with_config(cfg),
            model: model.into(),
            max_concurrency,
            cache,
        }
    }
}

#[async_trait::async_trait]
impl Llm for OpenAiCompatibleLlm {
    async fn chat_many(&self, prompts: Vec<Vec<ChatCompletionRequestMessage>>) -> Result<Vec<String>> {
        let requests = prompts.into_iter().enumerate().map(|(idx, messages)| {
            let client = self.client.clone();
            let model = self.model.clone();
            let cache = self.cache.clone();
            async move {
                let prompt_json = serde_json::to_string(&messages)?;
                let cache_key = cache.as_ref().map(|_| LlmCache::make_key(&model, &prompt_json));
                if let (Some(cache), Some(key)) = (cache.as_ref(), cache_key.as_deref()) {
                    if let Some(hit) = cache.get(key)? {
                        return Ok::<_, anyhow::Error>((idx, hit));
                    }
                }

                let req = CreateChatCompletionRequestArgs::default()
                    .model(model.clone())
                    .messages(messages)
                    .build()
                    .context("failed to build chat completion request")?;
                let resp = client.chat().create(req).await?;
                let text = resp.choices.first()
                    .and_then(|c| c.message.content.clone())
                    .unwrap_or_default();
                if let (Some(cache), Some(key)) = (cache.as_ref(), cache_key.as_deref()) {
                    cache.put(key, &text)?;
                }
                Ok::<_, anyhow::Error>((idx, text))
            }
        });

        let mut results = stream::iter(requests)
            .buffer_unordered(self.max_concurrency.max(1))
            .collect::<Vec<_>>()
            .await;

        results.sort_by_key(|item| item.as_ref().map(|(idx, _)| *idx).unwrap_or(usize::MAX));
        let mut out = Vec::with_capacity(results.len());
        for result in results {
            let (_, text) = result?;
            out.push(text);
        }
        Ok(out)
    }
}
