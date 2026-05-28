"""IAM-D inconsistency losses and optimizer."""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import Any, Callable, Iterable, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.func import functional_call

from .iam_utils import _disable_running_stats, _enable_running_stats

device = "cuda" if torch.cuda.is_available() else "cpu"


def inconsistencyLoss(model, image, label, criterion, beta, rho, noise_scale):
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    pred = functional_call(model, (params, buffers), (image,))
    pred_soft = F.softmax(pred, dim=1).clamp(min=1e-6, max=1.0)

    noise_norm = math.sqrt(sum(p.numel() for p in model.parameters() if p.requires_grad))
    noise_dict = {}
    for name, param in model.named_parameters():
        noise_dict[name] = (
            noise_scale * torch.normal(0, 1, size=param.data.shape, device=device) / noise_norm
        )
        param.data += noise_dict[name]
    noise_output = model(image)

    with torch.enable_grad():
        loss_kl = criterion_kl(F.log_softmax(noise_output, dim=1), pred_soft.detach())

    model.zero_grad()
    loss_kl.backward()
    grads = [param.grad.clone() for param in model.parameters() if param.requires_grad]
    wgrads = [torch.norm(param.grad, p=2) for param in model.parameters() if param.requires_grad]
    norm = torch.norm(torch.stack(wgrads), p=2) + 1e-12

    delta_dict = {}
    with torch.no_grad():
        for (name, param), grad in zip(model.named_parameters(), grads):
            delta_dict[name] = (rho * grad / norm).detach()
            param.data -= noise_dict[name]

    perturbed_params = {n: p + delta_dict[n] for (n, p) in params.items()}

    output_prime = functional_call(model, (perturbed_params, buffers), (image,))

    with precision_guard:
        p = pred_soft
        log_q = F.log_softmax(output_prime.float(), dim=1)
        inconsistency = F.kl_div(log_q, p, reduction="batchmean")

    loss = criterion(pred, label)

    return loss, beta * inconsistency


def inconsistency_FixMatch(
    model,
    labeled_images,
    labels,
    unlabeled_weak,
    strong_outputs,
    pseudo_labels,
    mask,
    criterion,
    criterion_u,
    lambda_u,
    scaler,
    beta,
    rho,
    noise_scale,
    *,
    amp_enabled: bool | None = None,
):
    if amp_enabled is None:
        amp_enabled = bool(scaler) and hasattr(scaler, "is_enabled") and scaler.is_enabled()
    precision_guard = (
        autocast(device_type="cuda", enabled=False)
        if (amp_enabled and device == "cuda")
        else nullcontext()
    )
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    model.train()
    outputs = model(labeled_images)
    loss_s = criterion(outputs, labels)
    loss_u = criterion_u(strong_outputs, pseudo_labels)
    mask_f = mask.float()
    loss_u = (loss_u * mask_f).sum() / mask_f.sum().clamp(min=1)

    image = torch.cat([labeled_images, unlabeled_weak], dim=0)
    with precision_guard:
        pred = functional_call(model, (params, buffers), (image,))
        pred_soft = F.softmax(pred, dim=1).clamp(min=1e-6, max=1.0)

    noise_norm = math.sqrt(sum(p.numel() for p in model.parameters() if p.requires_grad))
    noise_dict = {}
    for name, param in model.named_parameters():
        noise_dict[name] = (
            noise_scale * torch.normal(0, 1, size=param.data.shape, device=device) / noise_norm
        )
        param.data += noise_dict[name]

    noise_output = model(image)
    with precision_guard, torch.enable_grad():
        loss_kl = criterion_kl(F.log_softmax(noise_output.float(), dim=1), pred_soft.detach())

    model.zero_grad()
    loss_kl.backward()
    grads = [param.grad.clone() for param in model.parameters() if param.requires_grad]
    wgrads = [torch.norm(param.grad, p=2) for param in model.parameters() if param.requires_grad]
    norm = torch.norm(torch.stack(wgrads), p=2) + 1e-12

    delta_dict = {}
    with torch.no_grad(), precision_guard:
        for (name, param), grad in zip(model.named_parameters(), grads):
            delta_dict[name] = (rho * grad / norm).detach()
            param.data -= noise_dict[name]

    perturbed_params = {n: p + delta_dict[n] for (n, p) in params.items()}

    output_prime = functional_call(model, (perturbed_params, buffers), (image,))

    p = pred_soft
    log_q = F.log_softmax(output_prime, dim=1)
    inconsistency = F.kl_div(log_q, p, reduction="batchmean")

    loss = loss_s + lambda_u * loss_u

    return loss, beta * inconsistency


def inconsistency_semi(model, image, val_image, label, criterion, beta, rho, noise_scale):
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    ce_pred = functional_call(model, (params, buffers), (image,))
    val_pred = functional_call(model, (params, buffers), (val_image,))

    image = torch.cat([image, val_image], dim=0)

    pred = torch.cat([ce_pred, val_pred], dim=0)
    pred_soft = F.softmax(pred, dim=1).clamp(min=1e-6, max=1.0)

    noise_norm = math.sqrt(sum(p.numel() for p in model.parameters() if p.requires_grad))
    noise_dict = {}
    for name, param in model.named_parameters():
        noise_dict[name] = (
            noise_scale * torch.normal(0, 1, size=param.data.shape, device=device) / noise_norm
        )
        param.data += noise_dict[name]

    noise_output = model(image)

    with torch.enable_grad():
        loss_kl = criterion_kl(F.log_softmax(noise_output, dim=1), pred_soft.detach())

    model.zero_grad()
    loss_kl.backward()
    grads = [param.grad.clone() for param in model.parameters() if param.requires_grad]
    wgrads = [torch.norm(param.grad, p=2) for param in model.parameters() if param.requires_grad]
    norm = torch.norm(torch.stack(wgrads), p=2) + 1e-12

    delta_dict = {}
    with torch.no_grad():
        for (name, param), grad in zip(model.named_parameters(), grads):
            delta_dict[name] = (rho * grad / norm).detach()
            param.data -= noise_dict[name]

    perturbed_params = {n: p + delta_dict[n] for (n, p) in params.items()}

    output_prime = functional_call(model, (perturbed_params, buffers), (image,))

    p = pred_soft
    log_q = F.log_softmax(output_prime, dim=1)
    inconsistency = F.kl_div(log_q, p, reduction="batchmean")

    loss = criterion(ce_pred, label)

    return loss, beta * inconsistency


class SimclrLoss_IAM(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def get_logits(self, z_i, z_j, batch_size):
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = (z @ z.T) / (z.norm(dim=1)[:, None] * z.norm(dim=1)[None, :]) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)

        labels = torch.zeros(N, device=positive_samples.device, dtype=torch.int64)

        logits = torch.cat((positive_samples, negative_samples), dim=1)

        return logits, labels

    def forward(self, model, image, beta, rho, noise_scale):
        criterion_kl = nn.KLDivLoss(reduction="batchmean")

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        _, out_projection = functional_call(model, (params, buffers), (image,))
        batch_size = out_projection.shape[0] // 2
        p_out, p_distorted = torch.chunk(out_projection, 2, dim=0)

        logits, labels = self.get_logits(p_out, p_distorted, batch_size)

        pred_soft = F.softmax(logits, dim=1).clamp(min=1e-6, max=1.0)

        noise_norm = math.sqrt(sum(p.numel() for p in model.parameters() if p.requires_grad))
        noise_dict = {}
        for name, param in model.named_parameters():
            noise_dict[name] = (
                noise_scale * torch.normal(0, 1, size=param.data.shape, device=device) / noise_norm
            )
            param.data += noise_dict[name]
        _, noise_projection = model(image)

        noise_out, noise_distorted = torch.chunk(noise_projection, 2, dim=0)
        noise_logits, _ = self.get_logits(noise_out, noise_distorted, batch_size)

        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(noise_logits, dim=1), pred_soft.detach())

        model.zero_grad()
        loss_kl.backward()
        grads = [param.grad.clone() for param in model.parameters() if param.requires_grad]
        wgrads = [torch.norm(param.grad, p=2) for param in model.parameters() if param.requires_grad]
        norm = torch.norm(torch.stack(wgrads), p=2) + 1e-12

        delta_dict = {}
        with torch.no_grad():
            for (name, param), grad in zip(model.named_parameters(), grads):
                delta_dict[name] = (rho * grad / norm).detach()
                param.data -= noise_dict[name]

        perturbed_params = {n: p + delta_dict[n] for (n, p) in params.items()}

        _, projection_prime = functional_call(model, (perturbed_params, buffers), (image,))
        p_out_prime, p_distorted_prime = torch.chunk(projection_prime, 2, dim=0)
        logits_prime, _ = self.get_logits(p_out_prime, p_distorted_prime, batch_size)

        p = pred_soft
        log_q = F.log_softmax(logits_prime, dim=1)
        inconsistency = F.kl_div(log_q, p, reduction="batchmean")

        loss = self.criterion(logits, labels)
        loss /= batch_size * 2

        return loss, beta * inconsistency


__all__ = [
    "inconsistencyLoss",
    "inconsistency_FixMatch",
    "inconsistency_semi",
    "SimclrLoss_IAM",
    "IAM_D",
]

class IAM_D(torch.optim.Optimizer):
    r"""
    IAM-D (Direct Regularization) optimizer wrapper.

    Minimize:  L(θ) + β · max_{||δ||≤ρ}  (1/n) Σ KL( f(x;θ) || f(x;θ+δ) )
    - Inner (δ): K-step power-iteration-like ascent using gradient wrt (θ+δ).
      (KL inner에서는 p_ref(detached)를 타깃으로 사용)
    - Outer: one backward on L(θ) + β·KL(f(θ) || f(θ+δ*)) with BOTH-SIDE grads.
      (p_ref = softmax(f(θ))  **NO detach**)

    Closure contract:
      closure() -> (loss_main, logits_at_current_params)
      * closure는 backward를 호출하지 않아야 합니다.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer_cls: type,
        *,
        beta: float = 1.0,
        rho: float = 0.05,
        noise_scale: float = 1e-3,
        k_steps: int = 1,
        **base_opt_kwargs: Any,
    ):
        if beta < 0:
            raise ValueError("beta must be >= 0.")
        if rho <= 0:
            raise ValueError("rho must be > 0.")
        if k_steps < 1:
            raise ValueError("k_steps must be >= 1")

        defaults = dict(beta=beta, rho=rho, noise_scale=noise_scale, k_steps=k_steps)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_opt_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self._stateless_template: Optional[nn.Module] = None

    @torch.no_grad()
    def zero_grad(self, set_to_none: Optional[bool] = None) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def state_dict(self):
        return {"wrapper": super().state_dict(), "base_optimizer": self.base_optimizer.state_dict()}

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict["wrapper"])
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])

    def _collect_params_buffers(
        self, model: nn.Module
    ) -> Tuple[Dict[str, torch.nn.Parameter], Dict[str, torch.Tensor]]:
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        return params, buffers

    def _get_stateless_template(self, model: nn.Module) -> nn.Module:
        base_model = getattr(model, "module", model)
        if self._stateless_template is None:
            self._stateless_template = copy.deepcopy(base_model)
            device = next(base_model.parameters()).device
            self._stateless_template.to(device=device)
        self._stateless_template.train(base_model.training)
        return self._stateless_template

    @torch.no_grad()
    def _total_numel(self) -> int:
        n = 0
        for g in self.param_groups:
            for p in g["params"]:
                if p.requires_grad:
                    n += p.numel()
        return max(n, 1)

    def _iterable_params(self) -> List[torch.nn.Parameter]:
        return [p for g in self.param_groups for p in g["params"] if p.requires_grad]

    def step(
        self,
        closure: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
        model: nn.Module,
        inputs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one IAM-D step.

        Args:
            closure: () -> (loss_main, logits_at_w). Must NOT call backward().
            model:   nn.Module being optimized.
            inputs:  model inputs (tensor or tuple/list of tensors) used for logits at w+δ.

        Returns:
            (loss_main.detach(), (beta * KL).detach())
        """
        eps = 1e-12
        beta = self.defaults["beta"]
        rho = self.defaults["rho"]
        noise_scale = self.defaults["noise_scale"]
        k_steps = self.defaults["k_steps"]

        args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)

        # ------------------------------------------------------------
        # [Inner] Build p_ref (detached) at θ (no graph needed here)
        # ------------------------------------------------------------
        with torch.no_grad():
            logits_w_ng = model(*args)
            p_ref_detached = F.softmax(logits_w_ng, dim=1).clamp_min(torch.finfo(logits_w_ng.dtype).tiny)

        # ------------------------------------------------------------
        # [Inner] Estimate δ* via K-step normalized gradient ascent
        #         g_k = ∇_{δ} KL( f(θ) || f(θ+δ) ) at (θ+δ_k)
        #         δ_{k+1} = ρ * g_k / ||g_k||
        #         (autograd.grad wrt parameters at θ+δ_k; p_ref is detached)
        # ------------------------------------------------------------
        active_params = self._iterable_params()
        if not active_params:
            # No trainable params: nothing to do
            loss_main, logits_w = closure()
            return loss_main.detach(), torch.zeros((), device=logits_w.device)

        total_numel = self._total_numel()
        noise_std = noise_scale / math.sqrt(total_numel)

        # Use self.state to temporarily store deltas for the inner ascent
        for p in active_params:
            self.state[p]["inner_delta"] = torch.randn_like(p) * noise_std

        base_model = getattr(model, "module", model)
        params_dict, buffers_dict = self._collect_params_buffers(base_model)
        stateless_model = self._get_stateless_template(model)
        buffer_clones = {name: buf.detach().clone() for name, buf in buffers_dict.items()}

        _disable_running_stats(model)

        for _ in range(k_steps):
            # Build stateless perturbed weights (θ + δ_k) without mutating the original module.
            phantom_params: Dict[str, torch.Tensor] = {}
            phantom_tensors: List[torch.Tensor] = []
            
            for name, param in params_dict.items():
                if param.requires_grad:
                    delta = self.state[param]["inner_delta"]
                    base = param.detach()
                    base_plus_delta = (base + delta).requires_grad_(True)
                    phantom_params[name] = base_plus_delta
                    phantom_tensors.append(base_plus_delta)
                else:
                    phantom_params[name] = param.detach()

            with torch.enable_grad():
                noise_output = functional_call(stateless_model, (phantom_params, buffer_clones), args)
                log_q_noise = F.log_softmax(noise_output, dim=1)
                # KL using detached p_ref (inner ascent only)
                kl_inner = torch.mean(
                    torch.sum(p_ref_detached * (torch.log(p_ref_detached) - log_q_noise), dim=1)
                )

                grads = torch.autograd.grad(
                    kl_inner, phantom_tensors, create_graph=False, allow_unused=True
                )

            # g norm computed from grad tensors (float32 accumulation for stability)
            norm_sq = torch.zeros((), device=phantom_tensors[0].device, dtype=torch.float32)
            valid_grads = []
            valid_params = []
            for g, p_phantom in zip(grads, phantom_tensors):
                if g is not None:
                    gg = g.detach().reshape(-1).to(torch.float32)
                    norm_sq = norm_sq + gg.dot(gg)
                    valid_grads.append(g.detach())
                else:
                    valid_grads.append(None)
            
            g_norm = torch.sqrt(norm_sq + eps)
            scale = (rho / g_norm).to(phantom_tensors[0].device)
            
            # Update deltas in self.state
            grad_idx = 0
            for name, param in params_dict.items():
                if param.requires_grad:
                    g = valid_grads[grad_idx]
                    if g is not None:
                        self.state[param]["inner_delta"] = g * scale
                    grad_idx += 1

        _enable_running_stats(model)

        # ------------------------------------------------------------
        # [Outer] One backward on L(θ) + β·KL( f(θ) || f(θ+δ*) )
        #         BOTH-SIDE gradients: p_ref NOT detached here.
        #         Use stateless functional_call for θ+δ* to avoid in-place.
        # ------------------------------------------------------------
        self.base_optimizer.zero_grad()

        # Current-θ pass (with graph)
        loss_main, logits_w = closure()  # must not call backward

        # Prepare perturbed param dict θ+δ*
        perturbed_params: Dict[str, torch.Tensor] = {}
        for name, p in base_model.named_parameters():
            if p.requires_grad:
                d = self.state[p].get("inner_delta", None)
                if d is None:
                    perturbed_params[name] = p
                else:
                    perturbed_params[name] = (p + d).to(dtype=p.dtype)
            else:
                perturbed_params[name] = p

        # θ+δ forward (shares graph back to θ via (p + d))
        logits_delta = functional_call(stateless_model, (perturbed_params, buffer_clones), args)

        # KL with BOTH-SIDE grads
        p = F.softmax(logits_w, dim=1).clamp_min(torch.finfo(logits_w.dtype).tiny)
        log_p = F.log_softmax(logits_w, dim=1)
        log_q_delta = F.log_softmax(logits_delta, dim=1)
        kl_both = torch.mean(torch.sum(p * (log_p - log_q_delta), dim=1))  # KL(p||q)

        total_loss = loss_main + beta * kl_both
        total_loss.backward()
        self.base_optimizer.step()

        return loss_main.detach(), (beta * kl_both).detach()




class IAM_DE(torch.optim.Optimizer):
    r"""
    IAM-D (Direct Regularization) - Single-Backward Variant.

    목표:
      한 step에 forward 2번 (f(θ), f(θ+δ)) + backward 1번만 수행.
      backward에서 얻은 δ.grad로 δ를 power-iteration 1스텝(정규화된 ascent)처럼 갱신하여
      다음 스텝 penalty에 사용할 δ를 업데이트.

    목적함수:
      loss_total = L(θ) + β · KL( f(x;θ) || f(x;θ+δ) )
      - KL은 BOTH-SIDE gradient (p_ref = softmax(f(θ)) **detach 없이**)
      - δ는 requires_grad=True로 두어 동일 backward에서 δ.grad 획득

    δ 갱신(스텝 종료 시):
      g = ∇_δ [ KL(f(θ) || f(θ+δ)) ]  (같은 backward에서 획득)
      δ ← ρ · g / ||g||     (g=0이면 작은 노이즈로 재초기화)

    Closure 계약:
      closure() -> (loss_main, logits_at_w)  # backward 호출 금지

    사용법(기존 래퍼와 동일한 호출 형태):
      opt = IAM_D_OnePass(model.parameters(), torch.optim.SGD, lr=..., momentum=..., beta=1.0, rho=0.05)
      loss_main_detached, kl_term_detached = opt.step(closure, model, inputs)

    주의:
      - DDP 환경에서 δ는 rank별 독립 상태로 유지됩니다(요청 사항).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer_cls: type,
        *,
        beta: float = 1.0,
        rho: float = 0.05,
        noise_scale: float = 0.1,
        eps: float = 1e-12,
        **base_opt_kwargs: Any,
    ):
        if beta < 0:
            raise ValueError("beta must be >= 0.")
        if rho <= 0:
            raise ValueError("rho must be > 0.")

        defaults = dict(beta=beta, rho=rho, noise_scale=noise_scale, eps=eps)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_opt_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self._stateless_template: Optional[nn.Module] = None

    # --- 기본 위임/직렬화 ---
    @torch.no_grad()
    def zero_grad(self, set_to_none: Optional[bool] = None) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def state_dict(self):
        # self.state[param]에 δ가 저장되므로 super().state_dict()에 함께 포함됨
        return {"wrapper": super().state_dict(), "base_optimizer": self.base_optimizer.state_dict()}

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict["wrapper"])
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])

    # --- 내부 유틸 ---
    def _collect_params_buffers(
        self, model: nn.Module
    ) -> Tuple[Dict[str, torch.nn.Parameter], Dict[str, torch.Tensor]]:
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        return params, buffers

    def _get_stateless_template(self, model: nn.Module) -> nn.Module:
        base_model = getattr(model, "module", model)
        if self._stateless_template is None:
            self._stateless_template = copy.deepcopy(base_model)
            device = next(base_model.parameters()).device
            self._stateless_template.to(device=device)
        self._stateless_template.train(base_model.training)
        return self._stateless_template

    @torch.no_grad()
    def _total_numel(self) -> int:
        n = 0
        for g in self.param_groups:
            for p in g["params"]:
                if p.requires_grad:
                    n += p.numel()
        return max(n, 1)

    def _iterable_params(self) -> List[torch.nn.Parameter]:
        return [p for g in self.param_groups for p in g["params"] if p.requires_grad]

    @torch.no_grad()
    def reset_delta(self):
        """옵션: δ를 재초기화하고 싶을 때 호출."""
        for p in self._iterable_params():
            st = self.state[p]
            st.pop("delta", None)

    # --- 핵심 step ---
    def step(
        self,
        closure: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
        model: nn.Module,
        inputs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one optimization step with 2 forwards + 1 backward.

        Args:
            closure: () -> (loss_main, logits_at_w). Must NOT call backward().
            model:   nn.Module being optimized.
            inputs:  inputs for evaluating f(θ+δ) (tensor or tuple/list of tensors).

        Returns:
            (loss_main.detach(), (beta * KL).detach())
        """
        beta = self.defaults["beta"]
        rho = self.defaults["rho"]
        noise_scale = self.defaults["noise_scale"]
        eps = self.defaults["eps"]

        args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)

        # 활성 파라미터 수집
        active_params = self._iterable_params()
        if not active_params:
            loss_main, logits_w = closure()
            return loss_main.detach(), torch.zeros((), device=logits_w.device)

        # Randomize whether δ is added or subtracted this step.
        first_param = active_params[0]
        sign_factor = torch.tensor(1.0, device=first_param.device, dtype=first_param.dtype)
        if torch.randint(0, 2, (), device=first_param.device) == 0:
            sign_factor.neg_()

        total_numel = self._total_numel()
        noise_std = noise_scale / math.sqrt(total_numel)

        base_model = getattr(model, "module", model)
        params_dict, buffers_dict = self._collect_params_buffers(base_model)
        name_to_param = [(name, p) for name, p in base_model.named_parameters() if p.requires_grad]
        buffer_clones = {name: buf.detach().clone() for name, buf in buffers_dict.items()}
        stateless_model = self._get_stateless_template(model)

        # --- 1) 현재 θ에서 주 손실과 logits (closure는 backward 금지) ---
        #     여기서 그래프를 유지하여 KL의 BOTH-SIDE 그라디언트가 θ로 흐르도록 함.
        loss_main, logits_w = closure()

        # --- 2) θ+δ forward (δ는 requires_grad=True) ---
        # δ state 준비: 없으면 노이즈로 초기화, 있으면 그대로 사용
        delta_vars: Dict[str, torch.Tensor] = {}
        perturbed_params: Dict[str, torch.Tensor] = {}

        for name, p in params_dict.items():
            if not p.requires_grad:
                perturbed_params[name] = p
                continue

            st = self.state[p]
            d = st.get("delta", None)
            if d is None or d.shape != p.shape or d.device != p.device or d.dtype != p.dtype:
                d = torch.randn_like(p) * noise_std
                st["delta"] = d  # persist

            # 이번 step에서 autograd로 δ.grad를 받기 위해 복사본에 requires_grad 부여
            d_var = d.detach().clone().requires_grad_(True)
            delta_vars[name] = d_var

            # θ±δ (δ는 그래프 변수, 부호는 step마다 랜덤)
            perturbed_params[name] = (p + sign_factor * d_var).to(dtype=p.dtype)

        # BN 러닝스탯 오염 방지
        _disable_running_stats(model)
        logits_delta = functional_call(stateless_model, (perturbed_params, buffer_clones), args)
        _enable_running_stats(model)

        # --- 3) KL(p||q_delta) (BOTH-SIDE; p는 detach 안 함) ---
        p = F.softmax(logits_w, dim=1)
        tiny = torch.finfo(logits_w.dtype).tiny
        p = p.clamp_min(tiny)

        log_p = torch.log(p)
        log_q_delta = torch.log(F.softmax(logits_delta, dim=1).clamp_min(tiny))

        kl = torch.mean(torch.sum(p * (log_p - log_q_delta), dim=1))  # KL(p||q)

        # --- 4) total loss 한 번의 backward ---
        total_loss = loss_main + beta * kl
        self.base_optimizer.zero_grad()
        total_loss.backward()

        # --- 5) δ 갱신: g <- δ.grad 모아 정규화, δ ← ρ * g / ||g|| ---
        # grads 수집 및 노름 계산 (float32 누적)
        with torch.no_grad():
            # grads 목록 추출 (파라미터 순서대로)
            grads: List[torch.Tensor] = []
            for name, p in name_to_param:
                if name in delta_vars:
                    g = delta_vars[name].grad
                    if g is None:
                        # 사용되지 않은 파라미터일 수 있음: 0으로 처리
                        g = torch.zeros_like(delta_vars[name])
                    grads.append(g)

            if len(grads) == 0:
                g_norm = torch.tensor(0.0, device=logits_w.device)
            else:
                norm_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
                for g in grads:
                    gg = g.detach().reshape(-1).to(torch.float32)
                    norm_sq = norm_sq + gg.dot(gg)
                g_norm = torch.sqrt(norm_sq + eps)

            # 정규화 스케일
            if g_norm.item() <= 0.0 or not torch.isfinite(g_norm):
                # 거의 0 또는 NaN/Inf라면 노이즈로 재초기화
                for name, p in name_to_param:
                    if name in delta_vars:
                        self.state[p]["delta"] = torch.randn_like(p) * noise_std
            else:
                scale = (rho / g_norm).to(grads[0].device) if len(grads) > 0 else None
                # 파라미터별 δ state를 덮어쓰기
                gi = 0
                for name, p in name_to_param:
                    if name in delta_vars:
                        new_d = grads[gi].detach() * scale
                        # dtype/device 일관성
                        old_d = self.state[p]["delta"]
                        self.state[p]["delta"] = 0.9 * old_d + (1 - 0.9) * new_d.to(dtype=p.dtype, device=p.device)
                        gi += 1

        # --- 6) θ 업데이트 ---
        self.base_optimizer.step()

        return loss_main.detach(), (beta * kl).detach()


class IAM_DE_Momentum(torch.optim.Optimizer):
    r"""
    IAM-DE with Explicit Momentum and Extrapolation.

    This optimizer maintains a velocity vector 'delta_velocity' to track the 
    dominant eigenvector of the Hessian/FIM across batches, reducing noise.
    
    Args:
        delta_momentum (float): Momentum factor (mu) for the perturbation direction.
        delta_extrap (float): Nesterov extrapolation factor (tau). 
                              If > 0, applies correction: v_new + tau * (v_new - v_old).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer_cls: type,
        *,
        beta: float = 1.0,
        rho: float = 0.05,
        noise_scale: float = 1e-3, # Reduced noise as momentum stabilizes direction
        delta_momentum: float = 0.9,
        delta_extrap: float = 0.0, # Set to e.g., 0.5 or 1.0 for extrapolation
        eps: float = 1e-12,
        **base_opt_kwargs: Any,
    ):
        if beta < 0: raise ValueError("beta must be >= 0.")
        if rho <= 0: raise ValueError("rho must be > 0.")

        defaults = dict(
            beta=beta, 
            rho=rho, 
            noise_scale=noise_scale, 
            delta_momentum=delta_momentum,
            delta_extrap=delta_extrap,
            eps=eps
        )
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_opt_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self._stateless_template: Optional[nn.Module] = None

    @torch.no_grad()
    def zero_grad(self, set_to_none: Optional[bool] = None) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def state_dict(self):
        return {"wrapper": super().state_dict(), "base_optimizer": self.base_optimizer.state_dict()}

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict["wrapper"])
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])

    def _collect_params_buffers(self, model: nn.Module):
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        return params, buffers

    def _get_stateless_template(self, model: nn.Module) -> nn.Module:
        base_model = getattr(model, "module", model)
        if self._stateless_template is None:
            self._stateless_template = copy.deepcopy(base_model)
            device = next(base_model.parameters()).device
            self._stateless_template.to(device=device)
        self._stateless_template.train(base_model.training)
        return self._stateless_template

    @torch.no_grad()
    def _total_numel(self) -> int:
        n = 0
        for g in self.param_groups:
            for p in g["params"]:
                if p.requires_grad: n += p.numel()
        return max(n, 1)

    def _iterable_params(self) -> List[torch.nn.Parameter]:
        return [p for g in self.param_groups for p in g["params"] if p.requires_grad]

    def step(
        self,
        closure: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
        model: nn.Module,
        inputs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        beta = self.defaults["beta"]
        rho = self.defaults["rho"]
        noise_scale = self.defaults["noise_scale"]
        d_momentum = self.defaults["delta_momentum"]
        d_extrap = self.defaults["delta_extrap"]
        eps = self.defaults["eps"]

        args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
        active_params = self._iterable_params()
        if not active_params:
            loss_main, logits_w = closure()
            return loss_main.detach(), torch.zeros((), device=logits_w.device)

        # 1. Prepare Base Model & Inputs
        base_model = getattr(model, "module", model)
        params_dict, buffers_dict = self._collect_params_buffers(base_model)
        name_to_param = [(name, p) for name, p in base_model.named_parameters() if p.requires_grad]
        buffer_clones = {name: buf.detach().clone() for name, buf in buffers_dict.items()}
        stateless_model = self._get_stateless_template(model)

        total_numel = self._total_numel()
        noise_std = noise_scale / math.sqrt(total_numel)

        # 2. Main Loss (L(theta))
        loss_main, logits_w = closure()

        # 3. Prepare delta for forward pass (Use stored delta from previous step)
        delta_vars: Dict[str, torch.Tensor] = {}
        perturbed_params: Dict[str, torch.Tensor] = {}

        # Random sign flip for regularization robustness (optional but recommended in IAM-D)
        sign_factor = 1.0 if torch.rand(()) > 0.5 else -1.0

        for name, p in params_dict.items():
            if not p.requires_grad:
                perturbed_params[name] = p
                continue

            st = self.state[p]
            
            # Initialize delta if not present
            if "delta" not in st:
                d = torch.randn_like(p) * noise_std
                # Normalize initial noise to rho
                norm = d.norm() + eps
                d = d * (rho / norm)
                st["delta"] = d
                st["delta_velocity"] = d.clone() # Initialize velocity

            d = st["delta"]
            
            # Create graph-leaf for delta to get gradients
            d_var = d.detach().clone().requires_grad_(True)
            delta_vars[name] = d_var
            
            # Apply perturbation
            perturbed_params[name] = (p + sign_factor * d_var).to(dtype=p.dtype)

        # 4. Perturbed Forward (f(theta + delta))
        _disable_running_stats(model)
        logits_delta = functional_call(stateless_model, (perturbed_params, buffer_clones), args)
        _enable_running_stats(model)

        # 5. KL Divergence
        p_dist = F.softmax(logits_w, dim=1).clamp_min(eps)
        log_p = torch.log(p_dist)
        log_q = torch.log(F.softmax(logits_delta, dim=1).clamp_min(eps))
        kl = torch.mean(torch.sum(p_dist * (log_p - log_q), dim=1))

        # 6. Backward (Total Loss)
        total_loss = loss_main + beta * kl
        self.base_optimizer.zero_grad()
        total_loss.backward()

        # 7. Update Delta via Momentum & Extrapolation
        with torch.no_grad():
            # Collect raw gradients of delta
            grads = []
            for name, p in name_to_param:
                if name in delta_vars and delta_vars[name].grad is not None:
                    grads.append(delta_vars[name].grad)
                else:
                    grads.append(torch.zeros_like(p))
            
            # Calculate norm of the full gradient vector
            grad_norm_sq = sum(g.flatten().dot(g.flatten()) for g in grads)
            grad_norm = torch.sqrt(grad_norm_sq + eps)

            # Global update
            gi = 0
            for name, p in name_to_param:
                if name in delta_vars:
                    g = grads[gi]
                    st = self.state[p]
                    
                    # Normalized Gradient Direction
                    # (Prevent division by zero if grad is vanishing)
                    if grad_norm > 1e-8:
                        d_grad_normed = g / grad_norm
                    else:
                        d_grad_normed = torch.randn_like(g) * noise_std

                    # A. Momentum Update: v_{t+1} = mu * v_t + (1-mu) * g_normed
                    v_old = st["delta_velocity"]
                    v_new = d_momentum * v_old + (1 - d_momentum) * d_grad_normed
                    st["delta_velocity"] = v_new

                    # B. Extrapolation: u = v_new + tau * (v_new - v_old)
                    # This predicts the direction for the NEXT step
                    u = v_new + d_extrap * (v_new - v_old)

                    # C. Normalize u to rho to get next delta
                    # Note: We normalize 'u' individually per param group or globally? 
                    # IAM-D usually does global norm, but efficient implementation often does 
                    # renormalization at the end of the loop.
                    # Here we store 'u' temporarily, we need global norm of 'u' to strictly enforce rho.
                    # For efficiency in single-loop, we can approximate or do a second pass.
                    # Let's save 'u' to 'delta' directly but we need to normalize it globally.
                    st["delta"] = u 
                    gi += 1

            # Global Normalization of the new delta candidate
            # (Required to maintain constraint ||delta|| <= rho)
            delta_norm_sq = torch.tensor(0.0, device=logits_w.device)
            for p in active_params:
                d_cand = self.state[p]["delta"]
                delta_norm_sq += d_cand.flatten().dot(d_cand.flatten())
            
            delta_norm = torch.sqrt(delta_norm_sq + eps)
            scale = rho / delta_norm
            
            for p in active_params:
                self.state[p]["delta"].mul_(scale)

        # 8. Base Optimizer Step (theta update)
        self.base_optimizer.step()

        return loss_main.detach(), (beta * kl).detach()