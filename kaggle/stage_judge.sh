#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
JUDGE="${JUDGE:-google/gemini-3.6-flash}"
pick() { find /kaggle/input -path "*$1" | head -1; }
mkdir -p outputs/eval outputs/runs
for f in uf_dpo_vs_sft_judged.jsonl hh_dpo_vs_sft_judged.jsonl persona_lora_vs_base_judged.jsonl; do
  old="$(pick outputs/eval/$f)"
  if [ -n "$old" ] && [ ! -f outputs/eval/$f ]; then cp "$old" outputs/eval/$f; fi
done
for run in sft dpo_hh dpo_uf persona_lora; do
  src="$(dirname "$(pick outputs/$run/metrics.jsonl)")"
  mkdir -p outputs/runs/$run
  cp "$src"/*.json "$src"/metrics.jsonl outputs/runs/$run/
done
mkdir -p outputs/generations
for g in alpaca_base alpaca_sft alpaca_dpo_uf hh_base hh_sft hh_dpo_hh persona_base persona_lora sanity_base sanity_sft sanity_dpo_hh sanity_dpo_uf; do
  cp "$(pick outputs/generations/$g.json)" outputs/generations/
done
python scripts/build_pairs.py --target_file outputs/generations/alpaca_dpo_uf.json --anchor_file outputs/generations/alpaca_sft.json --track alpaca_eval --out_file outputs/eval/uf_dpo_vs_sft_pairs.jsonl
python scripts/build_pairs.py --target_file outputs/generations/hh_dpo_hh.json --anchor_file outputs/generations/hh_sft.json --track hh_eval --out_file outputs/eval/hh_dpo_vs_sft_pairs.jsonl
python scripts/build_pairs.py --target_file outputs/generations/persona_lora.json --anchor_file outputs/generations/persona_base.json --track persona_style --out_file outputs/eval/persona_lora_vs_base_pairs.jsonl
J="python scripts/judge_pairwise_kaggle.py --judge_model $JUDGE --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments"
$J --mode general --pairs_path outputs/eval/uf_dpo_vs_sft_pairs.jsonl --output_path outputs/eval/uf_dpo_vs_sft_judged.jsonl
$J --mode safety --pairs_path outputs/eval/hh_dpo_vs_sft_pairs.jsonl --output_path outputs/eval/hh_dpo_vs_sft_judged.jsonl
$J --mode persona --pairs_path outputs/eval/persona_lora_vs_base_pairs.jsonl --output_path outputs/eval/persona_lora_vs_base_judged.jsonl
python -c "import shutil; shutil.make_archive('final_artifacts', 'zip', '.', 'outputs')"
