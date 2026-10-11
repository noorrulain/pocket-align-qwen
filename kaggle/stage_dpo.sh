#!/usr/bin/env bash
set -euo pipefail
TRACK="$1"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" TOKENIZERS_PARALLELISM=false
bash kaggle/setup.sh
SFT_DIR="$(dirname "$(find /kaggle/input -path '*outputs/sft/best/config.json' | head -1)")"
echo "initial policy and frozen reference: $SFT_DIR"
if [ "$TRACK" = hh ]; then EVAL=eval_sets/hh_rlhf_eval_300_seed42.json; SET=hh; else EVAL=eval_sets/alpaca_eval_300_seed42.json; SET=alpaca; fi
mkdir -p outputs/generations
python scripts/prep_data.py --track "$TRACK"
python scripts/train_dpo.py --sft_ckpt "$SFT_DIR" --train_file data/prepared/${TRACK}_train.jsonl --val_file data/prepared/${TRACK}_val.jsonl --out_dir outputs/dpo_${TRACK}
python scripts/generate.py --model_path outputs/dpo_${TRACK}/best --generator qwen2.5-0.5b-dpo-${TRACK} --eval_file $EVAL --out_file outputs/generations/sanity_dpo_${TRACK}.json --limit 3 --show
python scripts/generate.py --model_path outputs/dpo_${TRACK}/best --generator qwen2.5-0.5b-dpo-${TRACK} --eval_file $EVAL --out_file outputs/generations/${SET}_dpo_${TRACK}.json
zip -qr dpo_${TRACK}_run_outputs.zip outputs/dpo_${TRACK}/*.json outputs/dpo_${TRACK}/*.jsonl outputs/generations
