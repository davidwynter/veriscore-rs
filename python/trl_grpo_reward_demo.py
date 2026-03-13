from __future__ import annotations

import os
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

from veriscore_reward_client import make_trl_reward_func


def build_demo_dataset() -> Dataset:
    rows = [
        {
            "prompt": "State two facts about aspirin relevant to hospital medication safety.",
            "prompt_source": "pharma",
            "k_median": 6,
        },
        {
            "prompt": "Explain what HbA1c measures and why clinicians track it.",
            "prompt_source": "clinical",
            "k_median": 5,
        },
        {
            "prompt": "Describe when a procurement invoice should match a purchase order.",
            "prompt_source": "procurement",
            "k_median": 4,
        },
        {
            "prompt": "Summarize two facts about bridge pier corrosion inspection intervals.",
            "prompt_source": "engineering",
            "k_median": 5,
        },
    ]
    return Dataset.from_list(rows)


def main() -> None:
    model_name = os.environ.get("GRPO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    reward_endpoint = os.environ.get("VERISCORE_REWARD_URL", "http://localhost:8088/grpo/reward_batch")
    output_dir = os.environ.get("GRPO_OUTPUT_DIR", "./outputs/grpo-veriscore-demo")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_demo_dataset()

    reward_func = make_trl_reward_func(
        endpoint=reward_endpoint,
        default_k_median=6,
        reward_metric="f1",
        binary=True,
    )

    config = GRPOConfig(
        output_dir=output_dir,
        run_name="grpo-veriscore-demo",
        learning_rate=1e-6,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=192,
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,
        bf16=True,
        beta=0.0,
        epsilon=0.2,
        num_iterations=1,
        loss_type="dr_grpo",
        scale_rewards="none",
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
