use anyhow::Result;
use async_openai::types::ChatCompletionRequestMessage;
use async_trait::async_trait;

pub struct FakeLlm {
    // a closure that maps each prompt to a JSON string
    pub handler: Box<dyn Fn(&[ChatCompletionRequestMessage]) -> String + Send + Sync>,
    pub delay_ms: u64,
}

#[async_trait]
impl veriscore_rs::llm::Llm for FakeLlm {
    async fn chat_many(&self, prompts: Vec<Vec<ChatCompletionRequestMessage>>) -> Result<Vec<String>> {
        use tokio::time::{sleep, Duration};
        let mut outs = Vec::with_capacity(prompts.len());
        for p in prompts.iter() {
            if self.delay_ms > 0 { sleep(Duration::from_millis(self.delay_ms)).await; }
            outs.push((self.handler)(p));
        }
        Ok(outs)
    }
}

pub struct FakeSearcher {
    pub results: Vec<veriscore_rs::serper::SerperItem>,
}

#[async_trait]
impl veriscore_rs::serper::Searcher for FakeSearcher {
    async fn search(&self, _query: &str) -> Result<Vec<veriscore_rs::serper::SerperItem>> {
        Ok(self.results.clone())
    }
}
