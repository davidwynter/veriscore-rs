// src/serper.rs
use anyhow::Result;
use governor::{Quota, RateLimiter};
use nonzero_ext::nonzero;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::time::Duration;

#[derive(Debug, Deserialize)]
pub struct SerperItem { pub title: String, pub link: String, pub snippet: String }

#[derive(Debug, Deserialize)]
struct SerperResp {
    #[serde(default)]
    organic: Vec<SerperItem>,
}

pub struct Serper {
    http: Client,
    key: String,
    limiter: RateLimiter,
    top_k: usize,
}

impl Serper {
    pub fn new(key: String, qps: u32, top_k: usize, timeout_ms: u64) -> Self {
        let http = Client::builder().timeout(Duration::from_millis(timeout_ms)).build().unwrap();
        let limiter = RateLimiter::direct(Quota::per_second(nonzero!(qps)));
        Self { http, key, limiter, top_k }
    }

    pub async fn search(&self, query: &str) -> Result<Vec<SerperItem>> {
        self.limiter.until_ready().await;
        let resp = self.http
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", &self.key)
            .json(&serde_json::json!({ "q": query, "num": self.top_k }))
            .send().await?
            .error_for_status()?
            .json::<SerperResp>().await?;
        Ok(resp.organic.into_iter().take(self.top_k).collect())
    }
}
