"""Training and evaluation loops used by the CLI entry points."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel

from .ddp_utils import DDPState, all_reduce_sum
from .optim_factory import OptimComponents
from torchvision.transforms import v2





_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

def _disable_running_stats(model: nn.Module) -> None:
    """배치 정규화 계층의 실행 통계 업데이트를 비활성화합니다."""
    def _disable(m: nn.Module):
        if isinstance(m, _BN_TYPES):
            if not hasattr(m, "backup_momentum"): # 모멘텀 값을 백업합니다 (첫 호출 시).
                m.backup_momentum = m.momentum
            m.momentum = 0.0 # 모멘텀을 0으로 설정하여 실행 통계 업데이트를 중지합니다.
    model.apply(_disable)

def _enable_running_stats(model: nn.Module) -> None:
    """비활성화된 배치 정규화 계층의 실행 통계 업데이트를 다시 활성화합니다."""
    def _enable(m: nn.Module):
        if isinstance(m, _BN_TYPES) and hasattr(m, "backup_momentum"):
            m.momentum = m.backup_momentum # 백업된 모멘텀 값으로 복원합니다.
    model.apply(_enable)

def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer_components: OptimComponents,
    ddp_state: DDPState,
    epoch: int,
    log_interval: int = 100,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> Dict[str, float]:

    model.train()


    sampler = getattr(train_loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    loss_original_sum = 0.0
    loss_s_sum = 0.0
    batches = 0

    optimizer = optimizer_components.optimizer
    scheduler = optimizer_components.scheduler
    scaler = optimizer_components.scaler
    use_amp = optimizer_components.use_amp
    is_step_scheduler = optimizer_components.is_step_scheduler

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batches += 1
        inputs = inputs.to(ddp_state.device, non_blocking=True)
        targets = targets.to(ddp_state.device, non_blocking=True)

        if mean is not None and std is not None:
             inputs = v2.functional.to_dtype(inputs, torch.float32, scale=True)
             inputs = v2.functional.normalize(inputs, mean, std)


        if optimizer_components.optimizer_name == "IAM_S":

            def closure():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                return loss, outputs.detach()

            loss_original, loss_s = optimizer.step(
                closure_main_loss=closure,
                model_nn_module=model,
                inputs_for_model=inputs,
            )

            if loss_original is not None:
                loss_original_sum += loss_original.item()
            if loss_s is not None:
                loss_s_sum += loss_s.item()

        elif optimizer_components.optimizer_name in ["IAM_D", "IAM_DE"]:

            def closure():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                return loss, outputs.detach()

            loss_original, _ = optimizer.step(
                closure,
                model,
                inputs,
            )

            if loss_original is not None:
                loss_original_sum += loss_original.item()

        elif optimizer_components.optimizer_name in ["SAM", "ASAM"]:
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            loss_original_sum += loss.item()
            
            _disable_running_stats(model)
            outputs_pertubed = model(inputs)
            loss_pertubed = criterion(outputs_pertubed, targets)
            loss_pertubed.backward()
            optimizer.second_step()
            _enable_running_stats(model)
            if loss_pertubed is not None:
                loss_s_sum += loss_pertubed.item()


        else:

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with autocast(device_type=ddp_state.device.type, dtype=torch.bfloat16, enabled=True):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
                
            loss_original_sum += loss.item()

        if is_step_scheduler and scheduler is not None:
            scheduler.step()

        if ddp_state.is_primary and batch_idx % log_interval == 0:
            if optimizer_components.optimizer_name == "IAM_S":
                print(
                    f"Epoch {epoch+1} Batch {batch_idx:04d}"
                    f" | Loss Original: {loss_original_sum / batches:.4f}"
                    f" | Loss S: {loss_s_sum / batches:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"Epoch {epoch+1} Batch {batch_idx:04d} | Loss: {loss_original_sum / batches:.4f}"
                    f" | LR: {optimizer.param_groups[0]['lr']:.6e} | StepSched: {is_step_scheduler}",
                    flush=True,
                )

    avg_original = loss_original_sum / max(batches, 1)
    avg_s = loss_s_sum / max(batches, 1)

    return {"loss_original": avg_original, "loss_s": avg_s}


@torch.no_grad()
def evaluate(model: nn.Module, data_loader, criterion: nn.Module, ddp_state: DDPState, mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,) -> Dict[str, float]:
    base_model = _unwrap_model(model)
    base_model.eval()

    top1_correct = 0
    top5_correct = 0
    total = 0
    eval_loss = 0.0

    for images, labels in data_loader:
        images = images.to(ddp_state.device, non_blocking=True)
        labels = labels.to(ddp_state.device, non_blocking=True)

        if mean is not None and std is not None:
            images = v2.functional.to_dtype(images, torch.float32, scale=True)
            images = v2.functional.normalize(images, mean, std)

        outputs = base_model(images)
        loss = criterion(outputs, labels)
        eval_loss += loss.item() * labels.size(0)

        _, predicted = outputs.max(1)
        top1_correct += predicted.eq(labels).sum().item()

        _, top5_predicted = outputs.topk(min(5, outputs.size(1)), 1, True, True)
        top5_correct += (top5_predicted == labels.view(-1, 1)).sum().item()
        total += labels.size(0)

    device = ddp_state.device
    metrics = torch.tensor([top1_correct, top5_correct, total, eval_loss], device=device, dtype=torch.float32)
    metrics = all_reduce_sum(metrics)

    top1 = metrics[0].item()
    top5 = metrics[1].item()
    total_samples = metrics[2].item()
    total_loss = metrics[3].item()

    top1_accuracy = 100.0 * top1 / total_samples if total_samples > 0 else 0.0
    top5_accuracy = 100.0 * top5 / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "eval_loss": avg_loss,
    }
