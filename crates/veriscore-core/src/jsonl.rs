use anyhow::{Context, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

pub fn read_jsonl<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<Vec<T>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("failed to open JSONL file {}", path.as_ref().display()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let item = serde_json::from_str::<T>(&line)
            .with_context(|| format!("invalid JSON on line {}", idx + 1))?;
        out.push(item);
    }
    Ok(out)
}

pub fn write_jsonl<T: Serialize, P: AsRef<Path>>(path: P, items: &[T]) -> Result<()> {
    let file = File::create(path.as_ref())
        .with_context(|| format!("failed to create JSONL file {}", path.as_ref().display()))?;
    let mut writer = BufWriter::new(file);
    for item in items {
        serde_json::to_writer(&mut writer, item)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}
