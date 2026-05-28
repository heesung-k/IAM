import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
torch.set_float32_matmul_precision('high')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training import (
    DDPEnvironment,
    build_distributed_dataloaders,
    build_model,
    build_optimizer_and_scheduler,
    evaluate,
    evaluate,
    train_one_epoch,
)
from data.imagenet.loaders import IMAGENET_MEAN, IMAGENET_STD
from data.cifar.datasets import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP training entry point for the IAM project.")
    parser.add_argument(
        "--dataset",
        default="imagenet",
        choices=["imagenet", "cifar10", "cifar100", "mnist", "fashion_mnist"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--optimizer",
        default="IAM_S",
        choices=["IAM_S", "SGD", "SAM", "ASAM", "IAM_D", "IAM_DE"],
        help="Optimizer to use.",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Global batch size across all devices.")
    parser.add_argument("--noise_scale", type=float, default=0.05, help="Noise scale for IAM optimizer.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability for WideResNet.")
    parser.add_argument("--rho", type=float, default=0.2, help="Perturbation radius for IAM optimizer.")
    parser.add_argument("--beta", type=float, default=1, help="Beta parameter for IAM_S optimizer.")
    parser.add_argument("--wd", type=float, default=None, help="Weight decay. If None, uses defaults.")
    parser.add_argument("--K", type=int, default=1, help="Number of inner steps for IAM_D optimizer.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate.")
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["linear", "cosine", "multistep"],
        help="LR scheduler type. If None, defaults based on model/dataset.",
    )
    parser.add_argument(
        "--model",
        default="resnet-50",
        choices=["resnet-18", "resnet-50", "WideResNet", "vit-s", "vit-t", "mlp_mixer-t"],
        help="Model architecture.",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for cross entropy.")
    parser.add_argument("--num_workers", type=int, default=6, help="DataLoader worker count per process.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/dataset/imagenet/",
        help="Root directory for ImageNet (ignored for torchvision datasets).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable Automatic Mixed Precision (supported for SGD).",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Batches before printing training metrics on the primary rank.",
    )
    parser.add_argument("--prefetch_factor", type=int, default=4,
                    help="Per-worker prefetch batches for DataLoader (num_workers>0).")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Use page-locked host memory for faster HtoD copies.")
    parser.add_argument("--pin_memory_device", type=str, default=None,
                        help='Target device for pinned memory, e.g., "cuda" or "cuda:0".')
    parser.add_argument("--persistent_workers", action="store_true",
                        help="Keep DataLoader workers alive across epochs.")
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Optional path to save the trained model checkpoint.",
    )
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for scheduler.")
    parser.add_argument("--min_lr", type=float, default=0.0, help="Minimum learning rate for scheduler.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    

    with DDPEnvironment() as ddp_state:
        if args.pin_memory_device is None and torch.cuda.is_available():
            args.pin_memory_device = f"cuda:{ddp_state.local_rank}"

        if ddp_state.is_primary:
            print("=" * 40)
            print(f"Optimizer: {args.optimizer}")
            print(f"World size: {ddp_state.world_size}")
            print(f"Dataset: {args.dataset} | Batch size (global): {args.batch_size}")
            print(f"Model: {args.model} | Rho: {args.rho} | K: {args.K} |Epochs: {args.epochs}")
            print(
                f"Learning rate: {args.lr} | Label smoothing: {args.label_smoothing}"
                f" | Noise scale: {args.noise_scale} | Dropout: {args.dropout}"
            )
            print(f"AMP enabled: {args.amp and args.optimizer != 'IAM_S'}")
            print("=" * 40)

        dataloaders = build_distributed_dataloaders(args, ddp_state)

        model = build_model(args, dataloaders.num_classes, dataloaders.input_channels, ddp_state.device)
        model = model.to(memory_format=torch.contiguous_format)
        if ddp_state.is_distributed:
            if ddp_state.device.type == "cuda":
                model = DDP(model, device_ids=[ddp_state.local_rank])
            else:
                model = DDP(model)
        model = torch.compile(model)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        optim_components = build_optimizer_and_scheduler(
            args, 
            model, 
            dataloaders.weight_decay,
            steps_per_epoch=len(dataloaders.train)
        )

        if ddp_state.is_primary:
            print(
                f"Per-device batch size: {dataloaders.per_device_batch_size}"
                f" | Global effective batch size: {dataloaders.per_device_batch_size * ddp_state.world_size}"
                f" | Steps per epoch: {len(dataloaders.train)}",
                flush=True,
            )

        start_time = time.perf_counter()

        if args.dataset == "imagenet":
            mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(ddp_state.device)
            std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(ddp_state.device)
        elif args.dataset == "cifar10":
            mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(ddp_state.device)
            std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(ddp_state.device)
        elif args.dataset == "cifar100":
            mean = torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1).to(ddp_state.device)
            std = torch.tensor(CIFAR100_STD).view(1, 3, 1, 1).to(ddp_state.device)
        else:
            mean = None
            std = None

        best_top1_error = 100.0

        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()

            train_metrics = train_one_epoch(
                model=model,
                train_loader=dataloaders.train,
                criterion=criterion,
                optimizer_components=optim_components,
                ddp_state=ddp_state,
                epoch=epoch,
                log_interval=args.log_interval,
                mean=mean,
                std=std,
            )

            # Scheduler step is now handled per-step in train_one_epoch
            # if not optim_components.is_step_scheduler:
            #    optim_components.scheduler.step()

            if ddp_state.device.type == "cuda":
                torch.cuda.synchronize(ddp_state.device)
            epoch_duration = time.perf_counter() - epoch_start

            eval_metrics = evaluate(model, dataloaders.eval, criterion, ddp_state, mean=mean, std=std)
            current_top1_error = 100.0 - eval_metrics['top1_accuracy']
            
            is_best = False
            if current_top1_error < best_top1_error:
                best_top1_error = current_top1_error
                is_best = True

            if ddp_state.is_primary:
                message = (
                    f"Epoch {epoch + 1}/{args.epochs}"
                    f" | Train Loss: {train_metrics['loss_original']:.4f}"
                    f" | Val Loss: {eval_metrics['eval_loss']:.4f}"
                    f" | Top-1 Error: {current_top1_error:.4f}%"
                    f" | Top-5 Error: {100.0 - eval_metrics['top5_accuracy']:.4f}%"
                    f" | Time: {epoch_duration:.2f}s"
                )
                if args.optimizer in ["IAM_S", "SAM", "ASAM"]:
                    message = message.replace(
                        "| Val Loss",
                        f"| Loss S: {train_metrics['loss_s']:.4f} | Val Loss",
                    )
                    
                if is_best and args.save_model:
                    save_path = Path(args.save_model)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    model_to_save = model.module if isinstance(model, DDP) else model
                    torch.save(
                        {
                            "model_state": model_to_save.state_dict(),
                            "args": vars(args),
                            "epoch": epoch,
                            "best_top1_error": best_top1_error
                        },
                        save_path,
                    )
                    message += " | [BEST SAVED]"
                    
                print(message, flush=True)

        total_time = time.perf_counter() - start_time
        if ddp_state.is_primary:
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            print(f"Training finished in {hours}h {minutes}m {seconds}s.")
            print(f"[{time.strftime('%H:%M:%S')}] Best Top-1 Error: {best_top1_error:.4f}%")

if __name__ == "__main__":
    main()
