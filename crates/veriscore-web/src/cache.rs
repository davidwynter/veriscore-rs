use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct WebCache {
    conn: Arc<Mutex<Connection>>,
}

impl WebCache {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS web_cache (
                cache_key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL
            );
            "#,
        )?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)) })
    }

    pub fn make_key(query: &str, top_k: usize) -> String {
        format!("{}:{}", top_k, format!("{:x}", md5::compute(query.as_bytes())))
    }

    pub fn get(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().expect("web cache poisoned");
        let mut stmt = conn.prepare("SELECT response_json FROM web_cache WHERE cache_key = ?1")?;
        let mut rows = stmt.query(params![key])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub fn put(&self, key: &str, response_json: &str) -> Result<()> {
        let conn = self.conn.lock().expect("web cache poisoned");
        conn.execute(
            "INSERT OR REPLACE INTO web_cache(cache_key, response_json) VALUES(?1, ?2)",
            params![key, response_json],
        )?;
        Ok(())
    }
}
