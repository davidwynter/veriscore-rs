use chrono::{DateTime, Utc};
use serde::Serialize;

pub fn stable_json_hash<T: Serialize>(value: &T) -> anyhow::Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(format!("{:x}", md5::compute(bytes)))
}

pub fn now_utc() -> DateTime<Utc> {
    Utc::now()
}

pub fn dedup_preserve_order<T: Eq + Clone>(items: &[T]) -> Vec<T> {
    let mut out = Vec::new();
    for item in items {
        if !out.contains(item) {
            out.push(item.clone());
        }
    }
    out
}
