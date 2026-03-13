use anyhow::Result;
use async_openai::types::ChatCompletionRequestMessage;

#[async_trait::async_trait]
pub trait Llm: Send + Sync {
    async fn chat_many(&self, prompts: Vec<Vec<ChatCompletionRequestMessage>>) -> Result<Vec<String>>;

    async fn chat_one(&self, prompt: Vec<ChatCompletionRequestMessage>) -> Result<String> {
        let mut out = self.chat_many(vec![prompt]).await?;
        out.pop().ok_or_else(|| anyhow::anyhow!("empty LLM response batch"))
    }
}
