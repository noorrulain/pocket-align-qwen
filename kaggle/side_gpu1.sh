#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ALP=eval_sets/alpaca_eval_300_seed42.json
HH=eval_sets/hh_rlhf_eval_300_seed42.json
STY=eval_sets/persona_style_prompts.json
QB=Qwen/Qwen2.5-0.5B
SM=HuggingFaceTB/SmolLM2-360M-Instruct
python scripts/generate.py --model_path $QB --generator qwen2.5-0.5b-base --eval_file $ALP --out_file outputs/generations/sanity_base.json --limit 3 --show
python scripts/generate.py --model_path $QB --generator qwen2.5-0.5b-base --eval_file $ALP --out_file outputs/generations/alpaca_base.json
python scripts/generate.py --model_path $QB --generator qwen2.5-0.5b-base --eval_file $HH --out_file outputs/generations/hh_base.json
python scripts/train_persona_lora.py --out_dir outputs/persona_lora
python scripts/generate.py --model_path $SM --style chat --generator smollm2-360m-instruct --eval_file $STY --out_file outputs/generations/persona_base.json --show
python scripts/generate.py --model_path $SM --adapter_path outputs/persona_lora/adapter --style chat --generator smollm2-360m-lora-dpo --eval_file $STY --out_file outputs/generations/persona_lora.json --show
