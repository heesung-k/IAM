"""Sharpness-aware minimisation losses and helpers."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from .iam_utils import _disable_running_stats, _enable_running_stats

def _grad_norm(parameters) -> torch.Tensor:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0, device=parameters[0].device if parameters else "cpu")
    return torch.norm(torch.stack([g.norm() for g in grads]))


def _param_backup(parameters):
    return [p.data.clone() for p in parameters if p.requires_grad]


def _iter_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def SAMLoss(model, image, label, criterion, optimizer, rho):
    params = _iter_params(model)

    pred = model(image)
    loss = criterion(pred, label)
    loss.backward()

    grads = [p.grad.detach().clone() for p in params]
    grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
    backup = _param_backup(params)

    with torch.no_grad():
        for param, grad in zip(params, grads):
            param.add_(rho * grad / (grad_norm + 1e-12))

    model.train()
    optimizer.zero_grad()
    pred_perturbed = model(image)
    loss_perturbed = criterion(pred_perturbed, label)
    loss_perturbed.backward()

    with torch.no_grad():
        for param, backup_param in zip(params, backup):
            param.data.copy_(backup_param)

    optimizer.step()
    optimizer.zero_grad()

    return loss_perturbed


def ASAMLoss(model, image, label, criterion, optimizer, rho):
    params = _iter_params(model)

    pred = model(image)
    loss = criterion(pred, label)
    loss.backward()

    grads = [p.grad.detach().clone() for p in params]
    grad_norm = torch.norm(
        torch.stack([
            ((torch.abs(param) + 0.01) * grad).norm(p=2)
            for param, grad in zip(params, grads)
        ])
    )

    backup = _param_backup(params)

    with torch.no_grad():
        for param, grad in zip(params, grads):
            scale = (param ** 2) + 0.01
            perturb = rho * grad / (grad_norm + 1e-12) * scale
            param.add_(perturb)

    model.train()
    optimizer.zero_grad()
    pred_perturbed = model(image)
    loss_perturbed = criterion(pred_perturbed, label)
    loss_perturbed.backward()

    with torch.no_grad():
        for param, backup_param in zip(params, backup):
            param.data.copy_(backup_param)

    optimizer.step()
    optimizer.zero_grad()

    return loss_perturbed


def FlatMatchLoss(
    model,
    labeled_images,
    labels,
    unlabeled_weak,
    unlabeled_strong,
    criterion,
    criterion_u,
    optimizer,
    rho,
    *,
    lambda_u=1,
    threshold=0.95,
):
    backup = _param_backup(_iter_params(model))

    with torch.no_grad():
        weak_outputs = model(unlabeled_weak)
        probs = torch.softmax(weak_outputs, dim=-1)
        max_probs, pseudo_labels = torch.max(probs, dim=-1)
        mask = max_probs.ge(threshold)

    optimizer.zero_grad()
    outputs = model(labeled_images)
    loss_s = criterion(outputs, labels)
    loss_s.backward(create_graph=True)

    params = _iter_params(model)
    grad_norm = _grad_norm(params)
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.data.add_(rho * p.grad / (grad_norm + 1e-12))

    logits_perturbed = model(unlabeled_strong)
    probs_perturbed = torch.softmax(logits_perturbed, dim=-1)
    with torch.no_grad():
        for param, backup_param in zip(params, backup):
            param.data.copy_(backup_param)
        logits_orig = model(unlabeled_strong)
        probs_orig = torch.softmax(logits_orig, dim=-1)
    cross_sharp_loss = F.kl_div(probs_perturbed.log(), probs_orig, reduction="batchmean")

    strong_outputs = model(unlabeled_strong)
    loss_u = criterion_u(strong_outputs, pseudo_labels)
    mask_f = mask.float()
    loss_u = (loss_u * mask_f).sum() / mask_f.sum().clamp(min=1)

    total_loss = loss_s + cross_sharp_loss + lambda_u * loss_u

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                # Adaptive SAM인 경우 p 자체도 FP32로 올려서 곱해야 함
                multiplier = (torch.pow(p.to(torch.float32), 2) if group["adaptive"] else 1.0)
                e_w = multiplier * p.grad.to(torch.float32) * scale
                # 가중치 업데이트 (여기서 다시 원래 dtype으로 캐스팅됨)
                p.add_(e_w.to(p.dtype))

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure, model, inputs):
        # (1) [First Step] 현재 위치(w)에서 Gradient 계산
        self.zero_grad()
        loss_original, _ = closure()
        loss_original.backward()
        
        # (2) [Ascent] 섭동(epsilon) 생성 및 적용 (w -> w + e)
        # zero_grad=True로 설정하여 w에서의 grad를 비움 (SAM 표준 절차)
        self.first_step(zero_grad=True)
        
        # (3) [Second Step] 섭동된 위치(w+e)에서 Gradient 계산
        _disable_running_stats(model) # 섭동 단계에서는 BN 통계 업데이트 방지
        loss_perturbed, _ = closure()
        loss_perturbed.backward()
        _enable_running_stats(model)
        
        # (4) [Descent] 원래 위치로 복귀 후 업데이트
        self.second_step()
        
        return loss_original.detach(), loss_perturbed.detach()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        (
                            (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                        ).to(dtype=torch.float32).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups



__all__ = ["SAMLoss", "ASAMLoss", "FlatMatchLoss"]
