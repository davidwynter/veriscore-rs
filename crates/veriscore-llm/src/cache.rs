use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct LlmCache {
    conn: Arc<Mutex<Connection>>,
}

impl LlmCache {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL
            );
            "#,
        )?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)) })
    }

    pub fn make_key(model: &str, prompt_json: &str) -> String {
        format!("{}:{}", model, format!("{:x}", md5::compute(prompt_json.as_bytes())))
    }

    pub fn get(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().expect("llm cache poisoned");
        let mut stmt = conn.prepare("SELECT response FROM llm_cache WHERE cache_key = ?1")?;
        let mut rows = stmt.query(params![key])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub fn put(&self, key: &str, response: &str) -> Result<()> {
        let conn = self.conn.lock().expect("llm cache poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache(cache_key, response) VALUES(?1, ?2)",
            params![key, response],
        )?;
        Ok(())
    }
}
