import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pocketalign.common import append_jsonl, dump_json, fix_seed, read_jsonl
from pocketalign.frame import frame_query
from pocketalign.objectives import dpo_step_terms, implicit_rewards
from pocketalign.packing import pack_pairs


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_ckpt", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--optimizer", choices=["rmsprop", "adamw"], default="rmsprop")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=32)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_batch", type=int, default=4)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy_device", default="cuda:0")
    ap.add_argument("--reference_device", default="cuda:1")
    return ap.parse_args()


def score_pairs(policy, reference, loader, beta, amp):
    policy.eval()
    gather = [[], [], [], []]
    with torch.no_grad():
        for pair_batch in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                _, parts = dpo_step_terms(policy, reference, pair_batch, beta)
            for slot, part in zip(gather, parts):
                slot.append(part.float().cpu())
    policy.train()
    pol_c, pol_r, ref_c, ref_r = (torch.cat(slot) for slot in gather)
    return pref_stats(pol_c, pol_r, ref_c, ref_r, beta)


def pref_stats(pol_c, pol_r, ref_c, ref_r, beta):
    win_r, lose_r = implicit_rewards(pol_c, pol_r, ref_c, ref_r, beta)
    edge = win_r - lose_r
    return {
        "loss": float(-torch.nn.functional.logsigmoid(edge).mean()),
        "reward_acc": float((win_r > lose_r).float().mean()),
        "reward_margin": float(edge.mean()),
        "chosen_reward": float(win_r.mean()),
        "rejected_reward": float(lose_r.mean()),
        "pairs": int(edge.numel()),
    }


def main():
    cfg = cli()
    fix_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    dump_json(os.path.join(cfg.out_dir, "run_config.json"), vars(cfg))
    amp = cfg.policy_device.startswith("cuda")
    tok = AutoTokenizer.from_pretrained(cfg.sft_ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    fit_rows, held_rows = read_jsonl(cfg.train_file), read_jsonl(cfg.val_file)
    if cfg.limit_train:
        fit_rows = fit_rows[: cfg.limit_train]
    if cfg.limit_val:
        held_rows = held_rows[: cfg.limit_val]
    policy = AutoModelForCausalLM.from_pretrained(cfg.sft_ckpt, torch_dtype=torch.float32, attn_implementation="sdpa").to(cfg.policy_device)
    reference = AutoModelForCausalLM.from_pretrained(cfg.sft_ckpt, torch_dtype=torch.float32, attn_implementation="sdpa").to(cfg.reference_device)
    policy.config.use_cache = False
    reference.config.use_cache = False
    reference.eval()
    reference.requires_grad_(False)

    def collate(rows):
        return pack_pairs(tok, [frame_query(r["query"]) for r in rows], [r["chosen"] for r in rows], [r["rejected"] for r in rows], cfg.max_length)

    shuffler = torch.Generator().manual_seed(cfg.seed)
    fit_loader = DataLoader(fit_rows, batch_size=cfg.micro_batch, shuffle=True, generator=shuffler, collate_fn=collate)
    held_loader = DataLoader(held_rows, batch_size=cfg.eval_batch, shuffle=False, collate_fn=collate)
    micro_total = len(fit_loader) * cfg.epochs
    update_total = math.ceil(micro_total / cfg.accum)
    if cfg.optimizer == "rmsprop":
        opt = torch.optim.RMSprop(policy.parameters(), lr=cfg.lr)
    else:
        opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    best_acc, best_step, update, micro = -1.0, 0, 0, 0
    bucket = [[], [], [], [], []]
    clock = time.time()
    policy.train()
    for epoch in range(cfg.epochs):
        for pair_batch in fit_loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss, parts = dpo_step_terms(policy, reference, pair_batch, cfg.beta)
            scaler.scale(loss / cfg.accum).backward()
            bucket[0].append(loss.detach().float().cpu().view(1))
            for slot, part in zip(bucket[1:], parts):
                slot.append(part.float().cpu())
            micro += 1
            if micro % cfg.accum and micro != micro_total:
                continue
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            update += 1
            stats = pref_stats(*(torch.cat(slot) for slot in bucket[1:]), cfg.beta)
            stats.update({"kind": "train", "step": update, "loss": float(torch.cat(bucket[0]).mean()), "grad_norm": float(gnorm), "pairs_seen": micro * cfg.micro_batch, "minutes": (time.time() - clock) / 60})
            bucket = [[], [], [], [], []]
            append_jsonl(log_path, stats)
            if update % 10 == 0:
                print(stats, flush=True)
            if update % cfg.eval_every == 0 or update == update_total:
                val = score_pairs(policy, reference, held_loader, cfg.beta, amp)
                val.update({"kind": "val", "step": update, "minutes": (time.time() - clock) / 60})
                append_jsonl(log_path, val)
                print(val, flush=True)
                if val["reward_acc"] > best_acc:
                    best_acc, best_step = val["reward_acc"], update
                    policy.save_pretrained(os.path.join(cfg.out_dir, "best"), safe_serialization=True)
                    tok.save_pretrained(os.path.join(cfg.out_dir, "best"))
    policy.save_pretrained(os.path.join(cfg.out_dir, "final"), safe_serialization=True)
    tok.save_pretrained(os.path.join(cfg.out_dir, "final"))
    dump_json(os.path.join(cfg.out_dir, "summary.json"), {"best_val_reward_acc": best_acc, "best_step": best_step, "updates": update, "reference_ckpt": cfg.sft_ckpt, "minutes": (time.time() - clock) / 60})


if __name__ == "__main__":
    main()
