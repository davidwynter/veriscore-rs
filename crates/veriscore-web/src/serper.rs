use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SerperItem {
    pub title: String,
    pub link: String,
    pub snippet: String,
}

#[derive(Debug, Deserialize)]
struct SerperResponse {
    #[serde(default)]
    organic: Vec<SerperItem>,
}

#[async_trait::async_trait]
pub trait Searcher: Send + Sync {
    async fn search(&self, query: &str) -> Result<Vec<SerperItem>>;
}

#[derive(Clone)]
pub struct Serper {
    http: reqwest::Client,
    api_key: String,
    default_top_k: usize,
}

impl Serper {
    pub fn new(http: reqwest::Client, api_key: String, default_top_k: usize) -> Self {
        Self {
            http,
            api_key,
            default_top_k,
        }
    }

    pub async fn search_impl(&self, query: &str) -> Result<Vec<SerperItem>> {
        let response = self.http
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", &self.api_key)
            .json(&serde_json::json!({
                "q": query,
                "num": self.default_top_k,
            }))
            .send()
            .await
            .context("failed to call Serper")?
            .error_for_status()
            .context("Serper returned non-success status")?;

        let parsed: SerperResponse = response
            .json()
            .await
            .context("failed to decode Serper response")?;

        Ok(parsed.organic)
    }
}

#[async_trait::async_trait]
impl Searcher for Serper {
    async fn search(&self, query: &str) -> Result<Vec<SerperItem>> {
        self.search_impl(query).await
    }
}