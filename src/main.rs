use clap::{Parser, Subcommand};
use anyhow::Result;
use veriscore_rs::*;

#[derive(Parser)]
#[command(name="veriscore", version)]
struct Cli {
  #[command(subcommand)]
  cmd: Cmd,
  /// Where to read/write JSONL
  #[arg(long, default_value="./data")] data_dir: String,
  /// Cache DB path
  #[arg(long, default_value="./data/cache.sqlite")] cache_db: String,
}

#[derive(Subcommand)]
enum Cmd {
  /// End-to-end: extract -> retrieve -> verify
  Run { #[arg(long)] input_file: String, #[arg(long, default_value="llama-3.3-70b-instruct")] extract_model: String, #[arg(long, default_value="llama-3.3-70b-instruct")] verify_model: String, #[arg(long, default_value_t=64)] llm_concurrency: usize, #[arg(long, default_value_t=16)] search_concurrency: usize, },
  Extract { #[arg(long)] input_file: String, #[arg(long, default_value="llama-3.3-70b-instruct")] model: String, #[arg(long, default_value_t=64)] llm_concurrency: usize },
  Retrieve { #[arg(long)] input_file: String, #[arg(long, default_value_t=16)] search_concurrency: usize, #[arg(long, default_value_t=10)] search_res_num: usize },
  Verify { #[arg(long)] input_file: String, #[arg(long, default_value="llama-3.3-70b-instruct")] model: String, #[arg(long, default_value_t=64)] llm_concurrency: usize, #[arg(long, default_value_t=1)] label_n: u8 },
  Score   { #[arg(long)] input_file: String, #[arg(long)] k_median: usize, #[arg(long, default_value="skip")] abstentions: String },
}

#[tokio::main]
async fn main() -> Result<()> {
  // init tracing...
  Ok(())
}
