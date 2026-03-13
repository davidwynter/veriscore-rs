use crate::traits::Llm;
use anyhow::{anyhow, Result};
use async_openai::types::ChatCompletionRequestMessage;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{sleep, Duration, Instant};
use tracing::{debug, warn};

#[derive(Debug, Clone)]
pub struct MicroBatchConfig {
    pub max_batch_size: usize,
    pub max_wait: Duration,
    pub queue_capacity: usize,
}

impl Default for MicroBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait: Duration::from_millis(10),
            queue_capacity: 1024,
        }
    }
}

struct BatchItem {
    prompt: Vec<ChatCompletionRequestMessage>,
    tx: oneshot::Sender<Result<String>>,
}

#[derive(Clone)]
pub struct BatchedLlm {
    tx: mpsc::Sender<BatchItem>,
}

impl BatchedLlm {
    pub fn spawn(inner: Arc<dyn Llm>, cfg: MicroBatchConfig) -> Self {
        let (tx, mut rx) = mpsc::channel::<BatchItem>(cfg.queue_capacity);
        tokio::spawn(async move {
            while let Some(first) = rx.recv().await {
                let mut batch = vec![first];
                let deadline = Instant::now() + cfg.max_wait;

                while batch.len() < cfg.max_batch_size {
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    let remaining = deadline - now;
                    tokio::select! {
                        biased;
                        maybe_item = rx.recv() => {
                            match maybe_item {
                                Some(item) => batch.push(item),
                                None => break,
                            }
                        }
                        _ = sleep(remaining) => break,
                    }
                }

                let prompts = batch.iter().map(|b| b.prompt.clone()).collect::<Vec<_>>();
                match inner.chat_many(prompts).await {
                    Ok(outputs) => {
                        if outputs.len() != batch.len() {
                            let err = anyhow!("batch size mismatch: got {} outputs for {} prompts", outputs.len(), batch.len());
                            for item in batch {
                                let _ = item.tx.send(Err(anyhow!(err.to_string())));
                            }
                        } else {
                            for (item, text) in batch.into_iter().zip(outputs) {
                                let _ = item.tx.send(Ok(text));
                            }
                        }
                    }
                    Err(err) => {
                        warn!(error = %err, "batched LLM call failed");
                        for item in batch {
                            let _ = item.tx.send(Err(anyhow!(err.to_string())));
                        }
                    }
                }
                debug!("flushed micro-batch");
            }
        });
        Self { tx }
    }

    pub async fn submit(&self, prompt: Vec<ChatCompletionRequestMessage>) -> Result<String> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(BatchItem { prompt, tx }).await.map_err(|_| anyhow!("micro-batcher closed"))?;
        rx.await.map_err(|_| anyhow!("micro-batch response channel closed"))?
    }
}

#[async_trait::async_trait]
impl Llm for BatchedLlm {
    async fn chat_many(&self, prompts: Vec<Vec<ChatCompletionRequestMessage>>) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            out.push(self.submit(prompt).await?);
        }
        Ok(out)
    }
}
