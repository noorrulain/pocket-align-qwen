import argparse
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from pocketalign.common import dump_json, fix_seed, load_json, write_jsonl


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--data_file", default="data/persona_lora/ramsay_style_preferences.json")
    ap.add_argument("--out_dir", default="outputs/persona_lora")
    ap.add_argument("--val_count", type=int, default=20)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def as_turns(rec):
    return {
        "prompt": [{"role": "user", "content": rec["prompt"]}],
        "chosen": [{"role": "assistant", "content": rec["chosen"]}],
        "rejected": [{"role": "assistant", "content": rec["rejected"]}],
    }


def main():
    cfg = cli()
    fix_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    dump_json(os.path.join(cfg.out_dir, "run_config.json"), vars(cfg))
    rows = load_json(cfg.data_file)
    order = list(range(len(rows)))
    random.Random(cfg.seed).shuffle(order)
    held = [as_turns(rows[i]) for i in order[: cfg.val_count]]
    fit = [as_turns(rows[i]) for i in order[cfg.val_count :]]
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32)
    adapter_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    knobs = DPOConfig(
        output_dir=os.path.join(cfg.out_dir, "trainer"),
        beta=cfg.beta,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=cfg.micro_batch,
        per_device_eval_batch_size=cfg.micro_batch,
        gradient_accumulation_steps=cfg.accum,
        num_train_epochs=cfg.epochs,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=cfg.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=knobs,
        train_dataset=Dataset.from_list(fit),
        eval_dataset=Dataset.from_list(held),
        processing_class=tok,
        peft_config=adapter_cfg,
    )
    trainer.train()
    trainer.model.save_pretrained(os.path.join(cfg.out_dir, "adapter"))
    tok.save_pretrained(os.path.join(cfg.out_dir, "adapter"))
    write_jsonl(os.path.join(cfg.out_dir, "metrics.jsonl"), trainer.state.log_history)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    dump_json(os.path.join(cfg.out_dir, "summary.json"), {"train_pairs": len(fit), "val_pairs": len(held), "trainable_params": trainable, "steps": trainer.state.global_step})


if __name__ == "__main__":
    main()
