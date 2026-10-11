import argparse
import random

from huggingface_hub import hf_hub_download

from pocketalign.common import dump_json, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_file", default="eval_sets/alpaca_eval_300_seed42.json")
    cfg = ap.parse_args()
    full = load_json(hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset", token=False))
    picks = random.Random(cfg.seed).sample(range(len(full)), cfg.count)
    dump_json(cfg.out_file, [{"id": f"alpaca-{cfg.seed}-{k:04d}", "source_index": i, "instruction": full[i]["instruction"], "dataset": "alpaca_eval", "source": full[i]["dataset"]} for k, i in enumerate(picks)])


if __name__ == "__main__":
    main()
