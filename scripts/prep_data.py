import argparse
import os

from transformers import AutoTokenizer

from pocketalign.common import write_jsonl
from pocketalign.sources import hh_rows, ultrachat_rows, ultrafeedback_rows


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["sft", "hh", "uf"], required=True)
    ap.add_argument("--out_dir", default="data/prepared")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--prompt_cap", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    cfg = cli()
    if cfg.track == "sft":
        fit = ultrachat_rows("train_sft", cfg.n_train or 30000, cfg.seed)
        held = ultrachat_rows("test_sft", cfg.n_val, cfg.seed)
    else:
        tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
        maker, splits = (hh_rows, ("train", "test")) if cfg.track == "hh" else (ultrafeedback_rows, ("train_prefs", "test_prefs"))
        fit = maker(splits[0], cfg.n_train or 15000, cfg.seed, tok, cfg.prompt_cap)
        held = maker(splits[1], cfg.n_val, cfg.seed, tok, cfg.prompt_cap)
    write_jsonl(os.path.join(cfg.out_dir, f"{cfg.track}_train.jsonl"), fit)
    write_jsonl(os.path.join(cfg.out_dir, f"{cfg.track}_val.jsonl"), held)
    print(cfg.track, "train", len(fit), "val", len(held))


if __name__ == "__main__":
    main()
