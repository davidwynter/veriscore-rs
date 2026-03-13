use crate::cache::WebCache;
use crate::serper::{SerperItem};
use anyhow::Result;
use std::sync::Arc;
use tokio::task::JoinSet;
use veriscore_core::types::EvidenceItem;

#[async_trait::async_trait]
pub trait EvidenceProvider: Send + Sync {
    async fn fetch_evidence_for_claims(&self, claims: &[String]) -> Result<Vec<(String, Vec<EvidenceItem>)>>;
}

#[derive(Clone)]
pub struct WebEvidenceProvider {
    serper: Arc<dyn crate::serper::Searcher>,
    top_k: usize,
    concurrency: usize,
    cache: Option<Arc<WebCache>>,
}

impl WebEvidenceProvider {
    pub fn new(serper: Arc<dyn crate::serper::Searcher>, top_k: usize, concurrency: usize, cache: Option<Arc<WebCache>>) -> Self {
        Self { serper, top_k, concurrency: concurrency.max(1), cache }
    }
}

#[async_trait::async_trait]
impl EvidenceProvider for WebEvidenceProvider {
    async fn fetch_evidence_for_claims(&self, claims: &[String]) -> Result<Vec<(String, Vec<EvidenceItem>)>> {
        let mut join_set = JoinSet::new();
        for claim in claims.iter().cloned() {
            let serper = self.serper.clone();
            let cache = self.cache.clone();
            let top_k = self.top_k;
            join_set.spawn(async move {
                let cache_key = cache.as_ref().map(|_| WebCache::make_key(&claim, top_k));
                if let (Some(cache), Some(key)) = (cache.as_ref(), cache_key.as_deref()) {
                    if let Some(hit) = cache.get(key)? {
                        let items: Vec<EvidenceItem> = serde_json::from_str(&hit)?;
                        return Ok::<_, anyhow::Error>((claim, items));
                    }
                }
                let raw = serper.search(&claim).await?;
                let items = raw.into_iter().map(to_evidence_item).take(top_k).collect::<Vec<_>>();
                if let (Some(cache), Some(key)) = (cache.as_ref(), cache_key.as_deref()) {
                    cache.put(key, &serde_json::to_string(&items)?)?;
                }
                Ok::<_, anyhow::Error>((claim, items))
            });
            if join_set.len() >= self.concurrency {
                let _ = join_set.join_next().await;
            }
        }

        let mut out = Vec::with_capacity(claims.len());
        while let Some(result) = join_set.join_next().await {
            out.push(result??);
        }
        out.sort_by_key(|(claim, _)| claims.iter().position(|c| c == claim).unwrap_or(usize::MAX));
        Ok(out)
    }
}

fn to_evidence_item(item: SerperItem) -> EvidenceItem {
    EvidenceItem {
        title: item.title,
        snippet: item.snippet,
        link: item.link,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serper::{Searcher, SerperItem};
    use std::sync::{Arc, Mutex};

    struct FakeSearcher {
        results: Vec<SerperItem>,
        calls: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl Searcher for FakeSearcher {
        async fn search(&self, query: &str) -> anyhow::Result<Vec<SerperItem>> {
            self.calls.lock().unwrap().push(query.to_string());
            Ok(self.results.clone())
        }
    }

    #[tokio::test]
    async fn fetch_evidence_for_claims_returns_results_for_each_claim() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let fake = FakeSearcher {
            results: vec![
                SerperItem {
                    title: "title1".to_string(),
                    link: "link1".to_string(),
                    snippet: "snippet1".to_string(),
                },
                SerperItem {
                    title: "title2".to_string(),
                    link: "link2".to_string(),
                    snippet: "snippet2".to_string(),
                },
            ],
            calls: calls.clone(),
        };

        let provider = WebEvidenceProvider::new(
            Arc::new(fake),
            2,
            8,
            None,
        );

        let claims = vec!["claim a".to_string(), "claim b".to_string()];
        let out = provider.fetch_evidence_for_claims(&claims).await.unwrap();

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, "claim a");
        assert_eq!(out[1].0, "claim b");
        assert_eq!(out[0].1.len(), 2);
        assert_eq!(out[0].1[0].title, "title1");
        assert_eq!(out[0].1[1].title, "title2");

        let call_log = calls.lock().unwrap();
        assert_eq!(call_log.len(), 2);
        assert!(call_log.contains(&"claim a".to_string()));
        assert!(call_log.contains(&"claim b".to_string()));
    }

    #[tokio::test]
    async fn fetch_evidence_for_claims_handles_empty_input() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let fake = FakeSearcher {
            results: vec![],
            calls: calls.clone(),
        };

        let provider = WebEvidenceProvider::new(
            Arc::new(fake),
            2,
            4,
            None,
        );

        let claims: Vec<String> = vec![];
        let out = provider.fetch_evidence_for_claims(&claims).await.unwrap();

        assert!(out.is_empty());
        assert_eq!(calls.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn fetch_evidence_for_claims_respects_top_k_if_provider_truncates() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let fake = FakeSearcher {
            results: vec![
                SerperItem {
                    title: "t1".to_string(),
                    link: "l1".to_string(),
                    snippet: "s1".to_string(),
                },
                SerperItem {
                    title: "t2".to_string(),
                    link: "l2".to_string(),
                    snippet: "s2".to_string(),
                },
                SerperItem {
                    title: "t3".to_string(),
                    link: "l3".to_string(),
                    snippet: "s3".to_string(),
                },
            ],
            calls,
        };

        let provider = WebEvidenceProvider::new(
            Arc::new(fake),
            2,
            4,
            None,
        );

        let claims = vec!["claim x".to_string()];
        let out = provider.fetch_evidence_for_claims(&claims).await.unwrap();

        assert_eq!(out.len(), 1);
        assert!(out[0].1.len() <= 2);
    }
}