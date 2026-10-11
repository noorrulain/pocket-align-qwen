#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python -m pip uninstall -y -q torchao || true
python -m pip install -q -r requirements.txt
python -m pip install -q -e . --no-deps
nvidia-smi
python -c "import torch, transformers, datasets, trl, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| datasets', datasets.__version__, '| trl', trl.__version__, '| peft', peft.__version__, '| visible gpus', torch.cuda.device_count())"
PYTHONPATH=. python -m pytest -q tests
