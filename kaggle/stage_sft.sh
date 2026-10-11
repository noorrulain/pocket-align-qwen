#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" TOKENIZERS_PARALLELISM=false
bash kaggle/setup.sh
mkdir -p outputs/generations outputs/logs
( CUDA_VISIBLE_DEVICES=1 bash kaggle/side_gpu1.sh > outputs/logs/side_gpu1.log 2>&1 || echo "side job failed" >> outputs/logs/side_gpu1.log ) &
SIDE=$!
python scripts/prep_data.py --track sft
python scripts/train_sft.py --out_dir outputs/sft --device cuda:0
python scripts/generate.py --model_path outputs/sft/best --generator qwen2.5-0.5b-sft --eval_file eval_sets/alpaca_eval_300_seed42.json --out_file outputs/generations/sanity_sft.json --limit 3 --show
python scripts/generate.py --model_path outputs/sft/best --generator qwen2.5-0.5b-sft --eval_file eval_sets/alpaca_eval_300_seed42.json --out_file outputs/generations/alpaca_sft.json
python scripts/generate.py --model_path outputs/sft/best --generator qwen2.5-0.5b-sft --eval_file eval_sets/hh_rlhf_eval_300_seed42.json --out_file outputs/generations/hh_sft.json
wait $SIDE || true
tail -n 80 outputs/logs/side_gpu1.log
python scripts/plot_runs.py
zip -qr sft_run_outputs.zip outputs/sft/*.json outputs/sft/*.jsonl outputs/persona_lora/*.json outputs/persona_lora/*.jsonl outputs/generations outputs/logs figures
