use anyhow::Result;
use async_openai::types::ChatCompletionRequestMessage;

#[async_trait::async_trait]
pub trait Llm: Send + Sync {
    async fn chat_many(&self, prompts: Vec<Vec<ChatCompletionRequestMessage>>) -> Result<Vec<String>>;
}
