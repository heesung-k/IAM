# Inconsistency-Aware Minimization (IAM)

Public research code for Inconsistency-Aware Minimization. This release keeps the
core optimizer implementations and the experiment entry points needed for
distributed supervised learning and FixMatch-style semi-supervised learning.

## Environment

This repository is managed with `uv` and currently targets Python `>=3.13.7`.

```bash
pip install -U uv
uv sync
```

## Entry Points

```bash
# Distributed supervised learning
uv run torchrun --nproc_per_node=4 experiments/train_ddp.py \
  --dataset cifar100 \
  --model WideResNet \
  --optimizer IAM_S

# Semi-supervised learning
uv run python experiments/semisup.py \
  --dataset CIFAR-10 \
  --optimizer IAM-D \
  --num_labeled 250
```

`experiments/train_ddp.py` is the maintained DDP supervised-learning launcher.
`experiments/semisup.py` is the maintained semi-supervised launcher; it is not
DDP-based.

## Data Paths

Dataset roots are resolved through CLI arguments where available. Shared data
utilities also support `IAM_DATA_DIR` and fall back to `/home/dataset/` for
legacy runs.

```bash
export IAM_DATA_DIR=/path/to/datasets
```

## Repository Layout

```text
.
├── data/
│   ├── cifar/                 # CIFAR datasets, loaders, and FixMatch splits
│   ├── imagenet/              # ImageNet datasets and loaders for DDP SL
│   ├── transforms/            # augmentation utilities
│   ├── config.py              # shared dataset-root resolution
│   ├── fashion_mnist.py
│   ├── registry.py
│   ├── svhn.py
│   └── utils.py
├── experiments/
│   ├── train_ddp.py           # DDP supervised learning
│   └── semisup.py             # semi-supervised learning
├── model/
│   ├── cnn.py
│   └── wideresnet.py
├── optim/
│   ├── iam.py
│   ├── iam_d.py
│   ├── iam_s.py
│   ├── iam_utils.py
│   └── sam.py
├── training/
│   ├── data_setup.py
│   ├── ddp_utils.py
│   ├── engine.py
│   ├── model_factory.py
│   └── optim_factory.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Public Release Notes

- This public tree intentionally omits auxiliary ablations, plotting scripts,
  tuning scripts, notebooks, logs, checkpoints, and exported figures.
- Add a top-level `LICENSE` file before publishing the repository.
- Review `wandb` defaults in `experiments/semisup.py` if runs should not log to
  a lab-specific entity by default.
