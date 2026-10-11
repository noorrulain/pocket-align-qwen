# pocket-align-qwen

SFT, from-scratch DPO (HH-RLHF and UltraFeedback) and TRL LoRA-DPO persona tuning for `Qwen/Qwen2.5-0.5B` and `HuggingFaceTB/SmolLM2-360M-Instruct`, run on Kaggle T4 x2.

Author: Noor Ul Ain Anwar (25280062)

## Layout

```text
pocketalign/            batching, response-only log-probs, SFT and DPO objectives, data loaders
scripts/                prep_data, train_sft, train_dpo, train_persona_lora, generate, build_pairs, judge_pairwise_kaggle, plot_runs
kaggle/                 one shell script per Kaggle notebook stage
eval_sets/              fixed HH eval set, AlpacaEval 300 (seed 42), persona style prompts
submission_adapter.py   grading interface
outputs/                metrics, configs, generations, pairwise files, judgments
figures/                training plots
report/                 final report
```

## Run on Kaggle (GPU T4 x2, internet on)

```bash
git clone https://github.com/noorrulain/pocket-align-qwen.git
bash pocket-align-qwen/kaggle/stage_sft.sh
bash pocket-align-qwen/kaggle/stage_dpo.sh hh
bash pocket-align-qwen/kaggle/stage_dpo.sh uf
bash pocket-align-qwen/kaggle/stage_judge.sh
PYTHONPATH=. python scripts/plot_runs.py
```

The DPO stages read the SFT checkpoint from the attached SFT notebook output. The judge stage runs in a Kaggle benchmark notebook with the earlier notebook outputs attached. The report is in `report/report.pdf`.

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests
```
