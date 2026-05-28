import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.amp import autocast, GradScaler

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data import get_fixmatch_loaders, get_fixmatch_loaders_cifar100
from model import WideResNet
from optim import inconsistency_FixMatch

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

@torch.no_grad()
def update_ema_model(src_model, ema_model, decay: float):
    # 1) 파라미터는 EMA
    for p_ema, p in zip(ema_model.parameters(), src_model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    # 2) 버퍼(BN running_mean/var, num_batches_tracked 등)는 '복사'
    for b_ema, b in zip(ema_model.buffers(), src_model.buffers()):
        b_ema.copy_(b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="IAM-D", type=str)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--ascent", default=0.05, type=float)
    parser.add_argument("--epochs", default=1024, type=int)
    parser.add_argument("--lr", default=0.03, type=float)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--dataset", default="CIFAR-10", type=str)
    parser.add_argument("--seed", default=5, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_labeled", default=250, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--prefetch_factor", default=2, type=int)
    parser.add_argument("--unlabeled_ratio", default=7, type=int)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.amp and device != "cuda":
        print("AMP requested but CUDA is unavailable; running in full precision.")
    amp_enabled = args.amp and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    wandb.init(
        entity="muwonijr-hanyang-university",
        project="IAM",
        # id="rd4e38wk",
        name=args.optimizer+"_FixMatch_EMA",
        config={
            "learning_rate": args.lr,
            "architecture": "WRN-28-2",
            "dataset": args.dataset,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "dropout": args.dropout,
            "ascent": args.ascent,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "num_labeled": args.num_labeled,
        })

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_labeled=args.num_labeled,
        seed=args.seed,
        prefetch_factor=args.prefetch_factor,
        unlabeled_ratio=args.unlabeled_ratio,
    )

    if args.dataset == "CIFAR-10":
        labeled_loader, unlabeled_loader, test_loader = get_fixmatch_loaders(**loader_kwargs)
        num_labels = 10
        channel = 3
    elif args.dataset == "CIFAR-100":
        labeled_loader, unlabeled_loader, test_loader = get_fixmatch_loaders_cifar100(**loader_kwargs)
        num_labels = 100
        channel = 3

    model = WideResNet(depth=28, width_factor=2, dropout=args.dropout, in_channels=channel, labels=num_labels).to(device)
    ema_model = WideResNet(depth=28, width_factor=2, dropout=args.dropout, in_channels=channel, labels=num_labels).to(device)
    ema_model.load_state_dict(model.state_dict())

    if args.torch_compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
        else:
            print("torch.compile is not available in this PyTorch build; continuing without compilation.")
    criterion = nn.CrossEntropyLoss()
    criterion_u = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=2**20)
    scaler = GradScaler(enabled=amp_enabled)

    # checkpoint_path = "./checkpoints/IAM-D_epoch500_seed1.pth"
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # start_epoch = checkpoint['epoch']

    steps_per_epoch = 1024
    lambda_u = 1
    threshold = 0.95
    ema_decay = 0.999
    ema_update_every = 4
    global_step = 0

    rho = args.ascent

    for epoch in range(args.epochs):
    # for epoch in range(start_epoch, args.epochs):
        model.train()
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        total_loss = 0.0
        total_inconsistency = 0.0
        chunk_timer = time.time()

        for step in range(steps_per_epoch):
            global_step += 1
            try:
                labeled_images, labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_images, labels = next(labeled_iter)
            try:
                (unlabeled_weak, unlabeled_strong), _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                (unlabeled_weak, unlabeled_strong), _ = next(unlabeled_iter)

            labeled_images, labels = labeled_images.to(device), labels.to(device)
            unlabeled_weak = unlabeled_weak.to(device)
            unlabeled_strong = unlabeled_strong.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                weak_outputs = model(unlabeled_weak)
                probs = torch.softmax(weak_outputs.float(), dim=-1)
                max_probs, pseudo_labels = torch.max(probs, dim=-1)
                mask = max_probs.ge(threshold)

            with autocast(device_type=autocast_device, enabled=amp_enabled):
                strong_outputs = model(unlabeled_strong)

                if args.optimizer == "SGD":
                    outputs = model(labeled_images)
                    loss_s = criterion(outputs, labels)
                    loss_u = criterion_u(strong_outputs, pseudo_labels)
                    mask_f = mask.float()
                    loss_u = (loss_u * mask_f).sum() / mask_f.sum().clamp(min=1)
                    loss = loss_s + lambda_u * loss_u
                elif args.optimizer == "IAM-D":
                    loss, inconsistency = inconsistency_FixMatch(
                        model, labeled_images, labels, unlabeled_weak,
                        strong_outputs, pseudo_labels, mask, criterion, criterion_u,
                        lambda_u, scaler, beta=args.beta, rho=rho, noise_scale=3.0,
                        amp_enabled=amp_enabled)
                    loss += inconsistency
                    total_inconsistency += inconsistency.detach().item()
                    # Clear gradients accumulated inside inconsistency_FixMatch before main backward
                    optimizer.zero_grad(set_to_none=True)
                else:
                    raise ValueError(f"Unsupported optimizer: {args.optimizer}")

            loss_value = loss.detach().item()
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()


            if global_step % ema_update_every == 0:
                corrected_decay = ema_decay ** ema_update_every
                update_ema_model(model, ema_model, corrected_decay)
            total_loss += loss_value
            scheduler.step()

            if (step + 1) % 128 == 0:
                chunk_duration = time.time() - chunk_timer
                wandb.log({"chunk_128_time_s": chunk_duration}, step=global_step)
                chunk_timer = time.time()
        
        avg_loss = total_loss / steps_per_epoch
        
        acc_model = evaluate(model)
        acc_ema   = evaluate(ema_model)
        wandb.log({"acc_model": acc_model, "acc_ema": acc_ema}, step=global_step)
        if epoch + 1 == 450:
            rho *= 0.5

        if (epoch + 1) % 50 == 0:
            if not os.path.exists("./checkpoints"):
                os.makedirs("./checkpoints")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"./checkpoints/{args.optimizer}_epoch{epoch+1}_seed{args.seed}_label{args.num_labeled}.pth")
            print(f"Model saved at epoch {epoch+1}")
