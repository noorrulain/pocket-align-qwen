import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from pocketalign.common import append_jsonl, dump_json, fix_seed, read_jsonl
from pocketalign.frame import frame_query
from pocketalign.objectives import span_logps
from pocketalign.packing import pack_rows


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train_file", default="data/prepared/sft_train.jsonl")
    ap.add_argument("--val_file", default="data/prepared/sft_val.jsonl")
    ap.add_argument("--out_dir", default="outputs/sft")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--micro_batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def held_out_loss(model, loader, amp):
    model.eval()
    nll, toks = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                total, count = span_logps(model, batch)
            nll -= total.sum().item()
            toks += count.sum().item()
    model.train()
    return nll / max(toks, 1)


def stash(model, tok, path):
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)


def main():
    cfg = cli()
    fix_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    dump_json(os.path.join(cfg.out_dir, "run_config.json"), vars(cfg))
    amp = cfg.device.startswith("cuda")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    fit_rows, held_rows = read_jsonl(cfg.train_file), read_jsonl(cfg.val_file)
    if cfg.limit_train:
        fit_rows = fit_rows[: cfg.limit_train]
    if cfg.limit_val:
        held_rows = held_rows[: cfg.limit_val]
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.float32, attn_implementation="sdpa").to(cfg.device)
    model.config.use_cache = False

    def collate(rows):
        return pack_rows(tok, [frame_query(r["query"]) for r in rows], [r["reply"] for r in rows], cfg.max_length)

    shuffler = torch.Generator().manual_seed(cfg.seed)
    fit_loader = DataLoader(fit_rows, batch_size=cfg.micro_batch, shuffle=True, generator=shuffler, collate_fn=collate)
    held_loader = DataLoader(held_rows, batch_size=cfg.eval_batch, shuffle=False, collate_fn=collate)
    micro_total = len(fit_loader) * cfg.epochs
    update_total = math.ceil(micro_total / cfg.accum)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    flat = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": flat, "weight_decay": 0.0}], lr=cfg.lr)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(cfg.warmup_frac * update_total)), update_total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    best_val, best_step, update, micro, bucket = float("inf"), 0, 0, 0, []
    clock = time.time()
    model.train()
    for epoch in range(cfg.epochs):
        for batch in fit_loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                total, count = span_logps(model, batch)
            loss = -total.sum() / count.sum().clamp_min(1)
            scaler.scale(loss / cfg.accum).backward()
            bucket.append(loss.item())
            micro += 1
            if micro % cfg.accum and micro != micro_total:
                continue
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            update += 1
            row = {"kind": "train", "step": update, "loss": sum(bucket) / len(bucket), "lr": sched.get_last_lr()[0], "grad_norm": float(gnorm), "examples": micro * cfg.micro_batch, "minutes": (time.time() - clock) / 60}
            bucket = []
            append_jsonl(log_path, row)
            if update % 10 == 0:
                print(row, flush=True)
            if update % cfg.eval_every == 0 or update == update_total:
                val = held_out_loss(model, held_loader, amp)
                append_jsonl(log_path, {"kind": "val", "step": update, "loss": val, "minutes": (time.time() - clock) / 60})
                print({"val_loss": val, "step": update}, flush=True)
                if val < best_val:
                    best_val, best_step = val, update
                    stash(model, tok, os.path.join(cfg.out_dir, "best"))
    stash(model, tok, os.path.join(cfg.out_dir, "final"))
    dump_json(os.path.join(cfg.out_dir, "summary.json"), {"best_val_loss": best_val, "best_step": best_step, "updates": update, "minutes": (time.time() - clock) / 60})


if __name__ == "__main__":
    main()
