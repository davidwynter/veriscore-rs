# VeriScore-rs: a fast, parallel Rust re-implementation of VeriScore

This reference implementation faithfully reproduces the VeriScore pipeline at https://arxiv.org/pdf/2508.05618 — **claim extraction → evidence retrieval → claim verification → scoring** — while building in the efficiency improvements from section 3.1.2 of https://arxiv.org/pdf/2508.05618: **full async concurrency, batched LLM requests, and non-blocking Serper evidence search**, with optional **Matrix (vLLM) backends** for high-throughput inference. The original VeriScore pipeline, inputs/outputs, and CLI knobs come from the upstream repo and paper; These are mirrored closely so you can swap between Python and Rust runs, compare outputs, and regression-test parity.

See REALTIME.md for an explanation of how veriscore-rs is used in combination with GRPO RL.

---

## Performance notes & how this hits “<5 seconds per response”

* **LLM throughput**: Use a **Matrix** cluster backed by vLLM; point `OPENAI_BASE_URL` to Matrix. Set `--llm-concurrency` high enough to saturate replicas (e.g., 64–256) and let **continuous batching** on the server do the rest. Enable **chunked prefill** on the server for large windows.
* **Evidence search**: With `qps` = 10–20 and `search_concurrency` \~64, Serper is typically sub-second per claim; the async pipeline keeps retrieval from dominating wall time.
* **I/O**: Streaming JSONL keeps memory bounded.
* **Caching**: The SQLite cache drastically reduces rerun times and cost: prompt-hash key → response body.
* **Binary verification**: Collapsing contradicted/inconclusive to unsupported (paper default) reduces label ambiguity and latency.

---

## Example runs

```bash
# 0) env
export SERPER_API_KEY=...
export OPENAI_BASE_URL=http://matrix.your.cluster/v1
export OPENAI_API_KEY=unused-or-per-matrix
# pick models served by Matrix:
export EXTRACT_MODEL="llama-3.3-70b-instruct"
export VERIFY_MODEL="llama-3.3-70b-instruct"

# 1) end-to-end
veriscore run --data_dir ./data --input_file samples.jsonl \
  --extract_model "$EXTRACT_MODEL" \
  --verify_model "$VERIFY_MODEL" \
  --llm_concurrency 128 --search_concurrency 64

# 2) stage-by-stage
veriscore extract  --data_dir ./data --input_file samples.jsonl --model "$EXTRACT_MODEL" --llm_concurrency 128
veriscore retrieve --data_dir ./data --input_file claims_samples.jsonl --search_concurrency 64 --search_res_num 10
veriscore verify   --data_dir ./data --input_file evidence_samples.jsonl --model "$VERIFY_MODEL" --llm_concurrency 128 --label_n 2

# 3) score (K = domain median)
veriscore score --data_dir ./data --input_file verification_samples.jsonl --k_median 8 --abstentions skip
```

---

## Design choices & deviations (all toggleable)

1. **Sentence segmentation:** I use `unicode-segmentation`’s UAX#29 sentence splitter: it’s lightweight and Unicode-correct. You can swap in a heavier rule-based segmenter if you need language-aware heuristics.
2. **Asynchrony:** LLM and Serper calls are async with back-pressure; concurrency is tuneable.
3. **Matrix/vLLM:** We rely on OpenAI-compatible APIs and let vLLM’s **continuous batching** do the hard work server-side; this is precisely where Matrix shines (replicas + autoscaling).
4. **Abstentions:** There’s an upstream question on whether to score abstentions as zero; I added `--abstentions=skip|zero`. The author indicated skipping is intended; pick what matches your study.
5. **Binary vs ternary labels:** Paper collapses ternary into binary for experiments. Keep ternary templates if you want more granular diagnostics.

---

## How this maps to the VeriScore paper & repo

* **Pipeline and outputs**: identical shapes/filenames, so your analysis notebooks can read either Python or Rust artifacts.
* **Extraction policy**: verifiable-only claims with sliding context; question is prepended for QA; first sentence used for long paragraphs (the windows mirror the paper’s description).
* **Evidence**: Serper `(title, snippet, link)` tuples per claim, as specified.
* **Verification**: supported vs unsupported (contradicted+inconclusive collapsed).
* **Scoring**: precision/recall with *K* as the domain median; domain-averaged VeriScore.

---

## Opinions / improvements beyond the original

* **Elastic micro-batcher:** In practice, pure fan-out (N concurrent calls) is more portable than vendor-specific “batch” endpoints — **and** works best with vLLM’s continuous batching. Keep concurrency high and let Matrix/vLLM batch server-side. (If your Matrix build exposes an explicit `/batch` endpoint, we can add a second transport that posts arrays.)
* **Chunked prefill at the server:** For long windows, enabling vLLM’s **chunked prefill** yields sizable throughput gains without changing client code.
* **Determinism for ablations:** Pin prompts to files; normalize whitespace; schema-validate LLM JSON with strict parsing; persist seeds in the server where possible.
* **Parallel tokenizer counts:** If you need `prompt_tok_cnt`/`response_tok_cnt`, call your server’s tokenization endpoint in parallel (Matrix/vLLM expose tokenization in many setups).

---

## Sources

* VeriScore repo & CLI shape; JSONL outputs; Serper requirement.
* VeriScore paper: pipeline, extraction policy, evidence schema, binary collapse, scoring with domain median *K*.
* Matrix: scalable LLM serving on top of vLLM; OpenAI-compatible APIs; autoscaling/load-balancing.
* vLLM docs: performance features like continuous batching and chunked prefill.
* Serper: endpoint and header usage.
* Llama-3.3-70B-Instruct (example model to serve via Matrix).

---

