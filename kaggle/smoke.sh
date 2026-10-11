#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" TOKENIZERS_PARALLELISM=false
T=/kaggle/tmp/smoke
mkdir -p $T
python scripts/prep_data.py --track sft --n_train 96 --n_val 16 --out_dir $T
python scripts/prep_data.py --track hh --n_train 96 --n_val 16 --out_dir $T
python scripts/prep_data.py --track uf --n_train 96 --n_val 16 --out_dir $T
python scripts/train_sft.py --train_file $T/sft_train.jsonl --val_file $T/sft_val.jsonl --out_dir $T/sft --accum 16 --eval_every 2
cat $T/sft/summary.json
python scripts/train_dpo.py --sft_ckpt $T/sft/best --train_file $T/uf_train.jsonl --val_file $T/uf_val.jsonl --out_dir $T/dpo --accum 32 --eval_every 2
cat $T/dpo/summary.json
python scripts/generate.py --model_path $T/dpo/best --generator smoke --eval_file eval_sets/hh_rlhf_eval_300_seed42.json --out_file $T/gen.json --limit 16 --max_new_tokens 128 --show
CUDA_VISIBLE_DEVICES=1 python scripts/train_persona_lora.py --out_dir $T/persona --epochs 0.3
CUDA_VISIBLE_DEVICES=1 python scripts/generate.py --model_path HuggingFaceTB/SmolLM2-360M-Instruct --adapter_path $T/persona/adapter --style chat --generator smoke-lora --eval_file eval_sets/persona_style_prompts.json --out_file $T/pgen.json --limit 2 --max_new_tokens 96 --show
python scripts/plot_runs.py --sft_dir $T/sft --hh_dir $T/dpo --uf_dir $T/none --persona_dir $T/persona --fig_dir $T/figs
ls $T/figs
