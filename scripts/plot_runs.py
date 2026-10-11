import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pocketalign.common import load_json, read_jsonl


def ema(vals, keep=0.9):
    out, run = [], None
    for v in vals:
        run = v if run is None else keep * run + (1 - keep) * v
        out.append(run)
    return out


def split(path):
    rows = read_jsonl(path)
    return [r for r in rows if r.get("kind") == "train"], [r for r in rows if r.get("kind") == "val"]


def best_step(run_dir):
    path = os.path.join(run_dir, "summary.json")
    return load_json(path).get("best_step") if os.path.exists(path) else None


def sft_fig(run_dir, fig_dir):
    fit, held = split(os.path.join(run_dir, "metrics.jsonl"))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([r["step"] for r in fit], [r["loss"] for r in fit], color="#9bb7d4", lw=0.7, label="train (raw)")
    ax.plot([r["step"] for r in fit], ema([r["loss"] for r in fit]), color="#1f4e79", lw=1.6, label="train (EMA 0.9)")
    ax.plot([r["step"] for r in held], [r["loss"] for r in held], "o-", color="#c0392b", ms=3, label="validation")
    pick = best_step(run_dir)
    if pick:
        ax.axvline(pick, ls="--", color="gray", lw=1, label=f"best ckpt (step {pick})")
    ax.set_xlabel("optimizer step (effective batch 32)")
    ax.set_ylabel("response-token NLL")
    ax.set_title("SFT on Qwen2.5-0.5B")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "sft_loss.png"), dpi=160)


def dpo_fig(runs, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    shades = {"HH-RLHF DPO": "#1f4e79", "UltraFeedback DPO": "#c0392b"}
    for name, run_dir in runs.items():
        path = os.path.join(run_dir, "metrics.jsonl")
        if not os.path.exists(path):
            continue
        fit, held = split(path)
        tint = shades[name]
        axes[0].plot([r["step"] for r in fit], ema([r["loss"] for r in fit]), color=tint, lw=1.2, label=f"{name} train (EMA 0.9)")
        axes[0].plot([r["step"] for r in held], [r["loss"] for r in held], "o--", color=tint, ms=3, label=f"{name} val")
        axes[1].plot([r["step"] for r in fit], ema([r["reward_acc"] for r in fit]), color=tint, lw=0.8, alpha=0.5, label=f"{name} train (EMA 0.9)")
        axes[1].plot([r["step"] for r in held], [r["reward_acc"] for r in held], "o-", color=tint, ms=3, label=f"{name} val")
        axes[2].plot([r["step"] for r in held], [r["reward_margin"] for r in held], "o-", color=tint, ms=3, label=f"{name} val")
        pick = best_step(run_dir)
        if pick:
            for ax in axes:
                ax.axvline(pick, ls=":", color=tint, lw=1)
    for ax, label in zip(axes, ("DPO loss", "reward accuracy", "reward margin")):
        ax.set_xlabel("optimizer step (effective batch 32 pairs)")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    axes[0].axhline(0.6931, color="gray", lw=0.8, ls="--")
    fig.suptitle("From-scratch DPO (dotted lines mark best-accuracy checkpoints)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dpo_curves.png"), dpi=160)


def persona_fig(run_dir, fig_dir):
    rows = read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    fit = [r for r in rows if "loss" in r and "eval_loss" not in r]
    held = [r for r in rows if "eval_loss" in r]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot([r["step"] for r in fit], [r["loss"] for r in fit], color="#1f4e79", label="train")
    axes[0].plot([r["step"] for r in held], [r["eval_loss"] for r in held], "o-", color="#c0392b", ms=3, label="validation")
    axes[0].set_ylabel("DPO loss")
    acc_key = "rewards/accuracies"
    axes[1].plot([r["step"] for r in fit if acc_key in r], [r[acc_key] for r in fit if acc_key in r], color="#1f4e79", label="train reward acc")
    axes[1].plot([r["step"] for r in held], [r.get("eval_" + acc_key) for r in held], "o-", color="#c0392b", ms=3, label="val reward acc")
    axes[1].plot([r["step"] for r in held], [r.get("eval_rewards/margins") for r in held], "s--", color="#7d3c98", ms=3, label="val reward margin")
    axes[1].set_ylabel("accuracy / margin")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.legend(fontsize=8)
    fig.suptitle("TRL DPOTrainer + LoRA on SmolLM2-360M-Instruct")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "persona_lora_curves.png"), dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_dir", default="outputs/sft")
    ap.add_argument("--hh_dir", default="outputs/dpo_hh")
    ap.add_argument("--uf_dir", default="outputs/dpo_uf")
    ap.add_argument("--persona_dir", default="outputs/persona_lora")
    ap.add_argument("--fig_dir", default="figures")
    cfg = ap.parse_args()
    os.makedirs(cfg.fig_dir, exist_ok=True)
    if os.path.exists(os.path.join(cfg.sft_dir, "metrics.jsonl")):
        sft_fig(cfg.sft_dir, cfg.fig_dir)
    dpo_fig({"HH-RLHF DPO": cfg.hh_dir, "UltraFeedback DPO": cfg.uf_dir}, cfg.fig_dir)
    if os.path.exists(os.path.join(cfg.persona_dir, "metrics.jsonl")):
        persona_fig(cfg.persona_dir, cfg.fig_dir)


if __name__ == "__main__":
    main()
