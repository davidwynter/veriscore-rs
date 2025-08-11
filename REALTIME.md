The Rust implementation already does low-latency, batched, async **extraction → retrieval → verification → scoring**. 
To make it usable *in-loop* for **GRPO** (Group Relative Policy Optimization), 
you expose a **batch reward API** that returns a scalar reward per completion fast enough to sit on the critical path of sampling → scoring → update. 
GRPO’s group-relative advantages are then computed on the trainer side from those rewards (e.g., reward minus the group mean). ([Verl][1], [Hugging Face][2], [finger-bone.github.io][3])

---

# GRPO training loop

## 1) Export a “reward microservice” from the Rust crate

Added a small Axum (or Actix) HTTP server to the Rust project:

* **Endpoint:** `POST /grpo/reward_batch`
* **Input:** one *prompt group* with **N** completions (N = GRPO group size), each completion is a `response` (plus optional `question/domain`).
* **Output:** vector of **rewards** (e.g., VeriScore-F1 or supported-fraction), plus optional diagnostics.

## Practical notes for *real-time* use

* **Batching/parallelism:** Keep `--llm-concurrency` high and let Matrix/vLLM do continuous batching server-side. Send **one HTTP batch per group** to amortize overhead.
* **Determinism/stability:** Fix the claim-extractor/ verifier prompts, set temperature to 0, and **cache** (SQLite) both LLM and Serper responses to reduce reward noise across epochs.
* **Throughput scaling:** Run multiple Rust reward replicas behind a load balancer; Matrix can scale LLM replicas independently (autoscaling and load-balancing are core Matrix features on top of vLLM). ([arXiv][4])
* **Reward shape:** Simple choices work well in practice:

  * **Supported-fraction**: `#supported_claims / #claims`
  * **F1\@K**: as in VeriScore (precision + “recall vs K median”) → map to `[0,1]`
    You can then standardize or clip before GRPO’s baseline subtraction.
* **Cost control:** Use short evidence lists (e.g., Serper top-5 or top-8), and deduplicate identical claims across the group before verifying.
* **Failure modes:** Timeouts from search or LLM should return a **neutral reward** (e.g., group mean) to avoid destabilizing updates.

---

## Limitations / caveats

* **Reward drift from web search:** Evidence changes over time. If you need strict stationarity during training, pin Serper parameters (`hl`, `gl`, `num`) and add a results cache snapshot per training run.
* **Binary vs ternary labels:** The original VeriScore often collapses “contradicted”/“inconclusive” into **unsupported** for evaluation; keep that default for reward stability. ([Hugging Face][2])
* **Privacy & egress:** If prompts are sensitive, run Serper-like retrieval on a private index instead of the public web.

---

## Bottom line

The Rust pipeline is already architected for **low-latency, batched, async** scoring. Adding a tiny HTTP/gRPC façade makes it a **drop-in, real-time reward function** for GRPO-style training in TRL/VERL/Unsloth. If you want, I can push a `veriscore-rewardd` binary with the Axum server, a Python client, and a GRPO demo notebook wired to TRL.

**Background on GRPO** (for completeness): GRPO is a group-based, critic-free variant of PPO used for LLM post-training: you sample multiple completions per prompt, compute relative advantages within the group, and optimize with a KL term to a reference policy. It has been popularized for reasoning-focused RL fine-tuning (e.g., DeepSeek-R1) and is supported in open-source trainers (TRL, VERL, Unsloth). ([Hugging Face][2], [Verl][1], [Unsloth Docs][5])

If you share your GRPO framework (TRL vs VERL vs custom) and target group size/throughput, I’ll provide the exact Axum routes and a small client wrapper tailored to your loop.

[1]: https://verl.readthedocs.io/en/latest/algo/grpo.html?utm_source=chatgpt.com "Group Relative Policy Optimization (GRPO) — verl documentation"
[2]: https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl?utm_source=chatgpt.com "Post training an LLM for reasoning with GRPO in TRL - Hugging Face"
[3]: https://finger-bone.github.io/rl-crashcourse/05/?utm_source=chatgpt.com "GRPO - Reinforcement Learning Crashcourse"
[4]: https://arxiv.org/abs/2405.20304?utm_source=chatgpt.com "Group Robust Preference Optimization in Reward-free RLHF"
[5]: https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo?utm_source=chatgpt.com "Tutorial: Train your own Reasoning model with GRPO | Unsloth Documentation"
