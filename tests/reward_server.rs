use axum::{Router, routing::post};
use serde_json::json;
use std::sync::Arc;
use veriscore_rs::server::{Engine};
use veriscore_rs::serper::{SerperItem, Searcher};
use veriscore_rs::llm::Llm;
use veriscore_rs::server;
use tower::ServiceExt; // for `oneshot`
use async_openai::types::ChatCompletionRequestMessage;

struct FakeExtract;
#[async_trait::async_trait]
impl Llm for FakeExtract {
    async fn chat_many(&self, _p: Vec<Vec<ChatCompletionRequestMessage>>) -> anyhow::Result<Vec<String>> {
        // Return a single claim per window; assume 1 window per completion for simplicity
        Ok(vec![r#"["Alpha claim"]"#.into()])
    }
}

struct FakeVerify;
#[async_trait::async_trait]
impl Llm for FakeVerify {
    async fn chat_many(&self, _p: Vec<Vec<ChatCompletionRequestMessage>>) -> anyhow::Result<Vec<String>> {
        Ok(vec![r#"{"label":"supported"}"#.into()])
    }
}

struct FakeSearch;
#[async_trait::async_trait]
impl Searcher for FakeSearch {
    async fn search(&self, _q: &str) -> anyhow::Result<Vec<SerperItem>> {
        Ok(vec![SerperItem{ title:"t".into(), link:"l".into(), snippet:"s".into() }])
    }
}

#[tokio::test]
async fn reward_batch_returns_rewards() {
    let engine = Engine {
        llm_extract: Arc::new(FakeExtract),
        llm_verify: Arc::new(FakeVerify),
        serper: Arc::new(FakeSearch),
        search_concurrency: 8,
        llm_concurrency: 8,
    };

    let app = Router::new()
        .route("/grpo/reward_batch", post(server::reward_batch))
        .with_state(Arc::new(engine));

    let payload = json!({
        "group_id":"g0",
        "k_median": 4,
        "binary": true,
        "completions": [
            {"question":"Q?","response":"Some generated text.","prompt_source":"general"},
            {"question":"Q?","response":"Another completion.","prompt_source":"general"}
        ]
    });

    let resp = app
        .oneshot(
            http::Request::post("/grpo/reward_batch")
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(serde_json::to_vec(&payload).unwrap().into())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), http::StatusCode::OK);
    let body = hyper::body::to_bytes(resp.into_body()).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(v.get("rewards").unwrap().as_array().unwrap().len() == 2);
    // With 1 supported claim out of 1, K=4 -> precision=1.0, recall=0.25, F1 ~ 0.4
    let r0 = v["rewards"][0].as_f64().unwrap();
    assert!(r0 > 0.39 && r0 < 0.41);
}
