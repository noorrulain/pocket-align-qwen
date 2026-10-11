import argparse
import random

from pocketalign.common import load_json, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_file", required=True)
    ap.add_argument("--anchor_file", required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--seed", type=int, default=42)
    cfg = ap.parse_args()
    target, anchor = load_json(cfg.target_file), load_json(cfg.anchor_file)
    anchor_by_id = {r["id"]: r for r in anchor}
    coin = random.Random(cfg.seed)
    rows = []
    for t in target:
        a = anchor_by_id[t["id"]]
        first, second = (t, a) if coin.random() < 0.5 else (a, t)
        rows.append({
            "pair_id": t["id"],
            "track": cfg.track,
            "category": t.get("category", ""),
            "instruction": t["instruction"],
            "response_a": first["output"],
            "response_b": second["output"],
            "hidden": {"a": first["generator"], "b": second["generator"], "target": t["generator"], "anchor": a["generator"]},
        })
    write_jsonl(cfg.out_file, rows)
    print(cfg.out_file, len(rows))


if __name__ == "__main__":
    main()
