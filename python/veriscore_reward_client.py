from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import asyncio
import aiohttp
import requests


@dataclass
class RewardConfig:
    endpoint: str = "http://localhost:8088/grpo/reward_batch"
    timeout_s: float = 60.0
    k_median: int = 8
    binary: bool = True
    reward_metric: str = "f1"


class VeriScoreRewardClient:
    def __init__(self, config: RewardConfig):
        self.config = config

    def score_batch(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        prompt_sources: Optional[Sequence[str]] = None,
        k_medians: Optional[Sequence[int]] = None,
        group_id: str = "group-0",
    ) -> Dict[str, Any]:
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have the same length")
        if prompt_sources is None:
            prompt_sources = ["general"] * len(prompts)
        if k_medians is None:
            if len(prompts) == 0:
                raise ValueError("empty batch")
            k_median = self.config.k_median
        else:
            if len(set(k_medians)) != 1:
                raise ValueError("current API expects a single k_median per batch; group by k_median first")
            k_median = int(k_medians[0])

        payload = {
            "group_id": group_id,
            "k_median": k_median,
            "binary": self.config.binary,
            "reward_metric": self.config.reward_metric,
            "completions": [
                {
                    "question": p,
                    "response": c,
                    "prompt_source": s,
                }
                for p, c, s in zip(prompts, completions, prompt_sources)
            ],
        }
        response = requests.post(self.config.endpoint, json=payload, timeout=self.config.timeout_s)
        response.raise_for_status()
        return response.json()

    async def score_batch_async(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        prompt_sources: Optional[Sequence[str]] = None,
        k_medians: Optional[Sequence[int]] = None,
        group_id: str = "group-0",
    ) -> Dict[str, Any]:
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have the same length")
        if prompt_sources is None:
            prompt_sources = ["general"] * len(prompts)
        if k_medians is None:
            if len(prompts) == 0:
                raise ValueError("empty batch")
            k_median = self.config.k_median
        else:
            if len(set(k_medians)) != 1:
                raise ValueError("current API expects a single k_median per batch; group by k_median first")
            k_median = int(k_medians[0])

        payload = {
            "group_id": group_id,
            "k_median": k_median,
            "binary": self.config.binary,
            "reward_metric": self.config.reward_metric,
            "completions": [
                {
                    "question": p,
                    "response": c,
                    "prompt_source": s,
                }
                for p, c, s in zip(prompts, completions, prompt_sources)
            ],
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.config.endpoint, json=payload) as response:
                response.raise_for_status()
                return await response.json()


def make_trl_reward_func(
    endpoint: str = "http://localhost:8088/grpo/reward_batch",
    timeout_s: float = 60.0,
    default_k_median: int = 8,
    reward_metric: str = "f1",
    binary: bool = True,
):
    """
    Build a TRL-compatible reward function.

    TRL passes `prompts`, `completions`, and dataset columns via `**kwargs`.
    This function supports:
      - standard format: prompts/completions are strings
      - conversational format: prompts/completions are message lists

    Expected optional dataset columns:
      - prompt_source
      - k_median
    """
    client = VeriScoreRewardClient(
        RewardConfig(
            endpoint=endpoint,
            timeout_s=timeout_s,
            k_median=default_k_median,
            binary=binary,
            reward_metric=reward_metric,
        )
    )

    def normalize_prompt(p: Any) -> str:
        if isinstance(p, str):
            return p
        if isinstance(p, list):
            # conversational format: flatten messages
            return "\n".join(m.get("content", "") for m in p if isinstance(m, dict))
        return str(p)

    def normalize_completion(c: Any) -> str:
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(m.get("content", "") for m in c if isinstance(m, dict))
        return str(c)

    def reward_func(prompts, completions, **kwargs):
        prompt_texts = [normalize_prompt(p) for p in prompts]
        completion_texts = [normalize_completion(c) for c in completions]
        prompt_sources = kwargs.get("prompt_source") or ["general"] * len(prompt_texts)
        k_medians = kwargs.get("k_median")

        group_id = kwargs.get("group_id")
        if group_id is None:
            group_id = f"trl-batch-{len(prompt_texts)}"

        data = client.score_batch(
            prompts=prompt_texts,
            completions=completion_texts,
            prompt_sources=prompt_sources,
            k_medians=k_medians,
            group_id=group_id,
        )
        return [float(x) for x in data["rewards"]]

    return reward_func


if __name__ == "__main__":
    cfg = RewardConfig()
    client = VeriScoreRewardClient(cfg)
    result = client.score_batch(
        prompts=["What is the capital of France?", "What is 2+2?"],
        completions=["Paris is the capital of France.", "The answer is 4."],
    )
    print(result)
