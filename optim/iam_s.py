"""IAM-S optimizer variants."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from .iam_utils import _check_finite, _disable_running_stats, _enable_running_stats

__all__ = ["IAM_S", "IAM_S_AMP", "IAM_S_ssl"]


class IAM_S(torch.optim.Optimizer):
    """
    제공된 훈련 로직을 통합한 커스텀 PyTorch 옵티마이저입니다.

    이 옵티마이저는 다음과 같은 단계로 작동합니다:
    1. 초기 손실을 계산합니다.
    2. 노이즈를 추가한 가중치에서 KL 발산 손실의 그래디언트(g)를 기반으로 섭동(delta_adv)을 계산합니다.
    3. 이 섭동을 원본 가중치에 적용한 후, 이 지점에서 새로운 손실(Loss_S) 및 그래디언트를 계산합니다.
    4. 원본 가중치를 복원하고, Loss_S에서 계산된 그래디언트를 사용하여 기본 옵티마이저 업데이트를 수행합니다.
    """

    def __init__(self, params, base_optimizer_cls, rho: float, noise_scale: float, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if noise_scale < 0.0:
            raise ValueError(f"Invalid noise_scale, should be non-negative: {noise_scale}")

        defaults = dict(rho=rho, noise_scale=noise_scale)
        super().__init__(params, defaults)

        self.k_val = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.k_val += p.numel()

        if self.k_val == 0:
            raise ValueError("Optimizer initialized with no trainable parameters.")

        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        if self.param_groups and self.param_groups[0]["params"]:
            self.device = self.param_groups[0]["params"][0].device
        else:
            self.device = torch.device("cpu")
            if self.k_val > 0:
                print(
                    "Warning: Optimizer has k_val > 0 but could not determine device from param_groups. "
                    "Defaulting to CPU."
                )

        self.criterion_kl = nn.KLDivLoss(reduction="batchmean", log_target=False).to(self.device)

    @torch.no_grad()
    def step(self, closure_main_loss, model_nn_module, inputs_for_model):
        if closure_main_loss is None:
            raise ValueError("closure_main_loss is required for this optimizer's step.")

        self.base_optimizer.zero_grad()

        loss_original, outputs_original = closure_main_loss()

        eps = torch.finfo(outputs_original.dtype).tiny if outputs_original.dtype.is_floating_point else 1e-8
        p_ref = F.softmax(outputs_original, dim=1).clamp_min(eps).detach()

        w_orig_vector = torch.nn.utils.parameters_to_vector(model_nn_module.parameters()).detach().clone()
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()

        _disable_running_stats(model_nn_module)

        d_noise = (
            self.defaults["noise_scale"]
            / math.sqrt(self.k_val + 1e-12)
            * torch.randn_like(w_orig_vector, device=self.device)
        )

        torch.nn.utils.vector_to_parameters(w_orig_vector + d_noise, model_nn_module.parameters())

        grad_g_vector = None
        with torch.enable_grad():
            logit_noise = model_nn_module(inputs_for_model)

            active_params_for_g = [p for p in model_nn_module.parameters() if p.requires_grad]
            if not active_params_for_g:
                _enable_running_stats(model_nn_module)
                print(
                    "Warning: No active parameters for 'g' calculation. Restoring original weights and skipping "
                    "perturbation."
                )
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                self.base_optimizer.step()
                return loss_original.detach(), None

            loss_noise_val = self.criterion_kl(F.log_softmax(logit_noise, dim=1), p_ref)

            if torch.isnan(loss_noise_val):
                _enable_running_stats(model_nn_module)
                print("Warning: NaN in loss_noise_val. Restoring original weights and skipping perturbation.")
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                self.base_optimizer.step()
                return loss_original.detach(), None

            grad_g_list = torch.autograd.grad(loss_noise_val, active_params_for_g, create_graph=False)

            flat_grad_g_list = [g.flatten() for g in grad_g_list if g is not None]
            if not flat_grad_g_list:
                _enable_running_stats(model_nn_module)
                print("Warning: Gradient 'g' is empty. Restoring original weights and using original gradients.")
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                self.base_optimizer.step()
                return loss_original.detach(), None
            grad_g_vector = torch.cat(flat_grad_g_list)

        delta_adv = self.defaults["rho"] * grad_g_vector / (grad_g_vector.norm(p=2) + 1e-12)
        del grad_g_vector

        _check_finite("delta_adv", delta_adv)

        torch.nn.utils.vector_to_parameters(w_orig_vector + delta_adv, model_nn_module.parameters())

        self.base_optimizer.zero_grad()
        with torch.enable_grad():
            loss_s, _ = closure_main_loss()
            loss_s.backward()

        _check_finite("loss_s", loss_s)

        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        _enable_running_stats(model_nn_module)

        return loss_original.detach(), loss_s.detach()

    @torch.no_grad()
    def step2(self, closure_main_loss, model_nn_module, inputs_for_model):
        if closure_main_loss is None:
            raise ValueError("closure_main_loss is required for this optimizer's step.")

        eps = 1e-12
        noise_scale = self.defaults["noise_scale"] / math.sqrt(self.k_val + eps)

        self.base_optimizer.zero_grad()
        loss_original, outputs_original = closure_main_loss()

        dl_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    dl_list.append(p.grad.data.clone().flatten())

        if not dl_list:
            print("Warning: No gradients (dl) found after initial closure. Skipping optimizer step.")
            return loss_original.detach() if loss_original is not None else None, None

        dl_vector = torch.cat(dl_list).detach()

        eps_val = torch.finfo(outputs_original.dtype).tiny if outputs_original.dtype.is_floating_point else 1e-8
        p_ref = F.softmax(outputs_original, dim=1).clamp_min(eps_val).detach()

        w_orig_vector = torch.nn.utils.parameters_to_vector(model_nn_module.parameters()).detach().clone()
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()

        _disable_running_stats(model_nn_module)

        d_noise = noise_scale * torch.randn_like(w_orig_vector, device=self.device)

        torch.nn.utils.vector_to_parameters(w_orig_vector + d_noise, model_nn_module.parameters())

        grad_g_vector = None
        with torch.enable_grad():
            logit_noise, projection_noise = model_nn_module(inputs_for_model)

            active_params_for_g = [p for p in model_nn_module.parameters() if p.requires_grad]
            if not active_params_for_g:
                _enable_running_stats(model_nn_module)
                print(
                    "Warning: No active parameters for 'g' calculation. Restoring original weights and skipping "
                    "perturbation."
                )
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                self.base_optimizer.step()
                return loss_original.detach(), None

            loss_noise_val = self.criterion_kl(F.log_softmax(projection_noise, dim=1), p_ref)

            if torch.isnan(loss_noise_val):
                _enable_running_stats(model_nn_module)
                print("Warning: NaN in loss_noise_val. Restoring original weights and skipping perturbation.")
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                self.base_optimizer.step()
                return loss_original.detach(), None

            grad_g_list = torch.autograd.grad(loss_noise_val, active_params_for_g, create_graph=False)

            flat_grad_g_list = [g.flatten() for g in grad_g_list if g is not None]
            if not flat_grad_g_list:
                _enable_running_stats(model_nn_module)
                print("Warning: Gradient 'g' is empty. Restoring original weights and using original gradients.")
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                self.base_optimizer.step()
                return loss_original.detach(), None
            grad_g_vector = torch.cat(flat_grad_g_list)

        delta_adv = self.defaults["rho"] * grad_g_vector / (grad_g_vector.norm(p=2) + 1e-12)

        if dl_vector.numel() == delta_adv.numel():
            delta_adv = torch.sign(torch.dot(delta_adv, dl_vector)) * delta_adv
        else:
            print(
                f"Warning: dl_vector size ({dl_vector.numel()}) and delta_adv (from g) size "
                f"({delta_adv.numel()}) mismatch. Skipping sign alignment for delta."
            )

        torch.nn.utils.vector_to_parameters(w_orig_vector + delta_adv, model_nn_module.parameters())

        self.base_optimizer.zero_grad()
        with torch.enable_grad():
            loss_s, _ = closure_main_loss()

        torch.nn.utils.clip_grad_norm_(model_nn_module.parameters(), max_norm=0.5)

        if torch.isnan(loss_s):
            print("NaN detected in Loss_S at step. Optimizer will use these NaN gradients from Loss_S.")

        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        _enable_running_stats(model_nn_module)

        return loss_original.detach(), loss_s.detach()


class IAM_S_AMP(torch.optim.Optimizer):
    """
    Custom PyTorch optimizer integrated with the provided training logic, made compatible with AMP.
    """

    def __init__(self, params, base_optimizer_cls, rho: float, noise_scale: float, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if noise_scale < 0.0:
            raise ValueError(f"Invalid noise_scale, should be non-negative: {noise_scale}")

        defaults = dict(rho=rho, noise_scale=noise_scale)
        super().__init__(params, defaults)

        self.k_val = sum(p.numel() for group in self.param_groups for p in group["params"] if p.requires_grad)
        if self.k_val == 0:
            raise ValueError("Optimizer initialized with no trainable parameters.")

        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        if not self.param_groups or not self.param_groups[0]["params"]:
            self.device = torch.device("cpu")
            print("Warning: Could not determine device from param_groups. Defaulting to CPU.")
        else:
            self.device = self.param_groups[0]["params"][0].device

        self.criterion_kl = nn.KLDivLoss(reduction="batchmean", log_target=False).to(self.device)

    @torch.no_grad()
    def step(self, closure_main_loss, model_nn_module, inputs_for_model, scaler: GradScaler):
        if closure_main_loss is None or scaler is None:
            raise ValueError("closure_main_loss and scaler are required for this optimizer's step.")

        with autocast(device_type="cuda"):
            loss_original, outputs_original = closure_main_loss()
        _check_finite("loss orig", loss_original)
        eps = torch.finfo(outputs_original.dtype).tiny if outputs_original.dtype.is_floating_point else 1e-8
        p_ref = F.softmax(outputs_original, dim=1).clamp_min(eps).detach()
        _check_finite("p_ref", p_ref)
        w_orig_vector = torch.nn.utils.parameters_to_vector(model_nn_module.parameters()).detach().clone()

        _disable_running_stats(model_nn_module)

        d_noise = (
            self.defaults["noise_scale"]
            / math.sqrt(self.k_val + 1e-12)
            * torch.randn_like(w_orig_vector, device=self.device)
        )

        torch.nn.utils.vector_to_parameters(w_orig_vector + d_noise, model_nn_module.parameters())

        with torch.enable_grad():
            with autocast(device_type="cuda"):
                logit_noise, projection_noise = model_nn_module(inputs_for_model)

            active_params_for_g = [p for p in model_nn_module.parameters() if p.requires_grad]
            if not active_params_for_g:
                _enable_running_stats(model_nn_module)
                print(
                    "Warning: No active parameters for 'g' calculation. Restoring original weights and skipping "
                    "perturbation."
                )
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                scaler.step(self.base_optimizer)
                return loss_original.detach(), None

            loss_noise_val = self.criterion_kl(F.log_softmax(projection_noise, dim=1), p_ref)

            if torch.isnan(loss_noise_val):
                _enable_running_stats(model_nn_module)
                print("Warning: NaN in loss_noise_val. Restoring original weights and skipping perturbation.")
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                scaler.step(self.base_optimizer)
                return loss_original.detach(), None

            grad_g_list = torch.autograd.grad(loss_noise_val, active_params_for_g, create_graph=False)

            flat_grad_g_list = [g.flatten() for g in grad_g_list if g is not None]
            if not flat_grad_g_list:
                _enable_running_stats(model_nn_module)
                print("Warning: Gradient 'g' is empty. Restoring original weights and using original gradients.")
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                scaler.step(self.base_optimizer)
                return loss_original.detach(), None
            grad_g_vector = torch.cat(flat_grad_g_list)

        delta_adv = self.defaults["rho"] * grad_g_vector / (grad_g_vector.norm(p=2) + 1e-12)
        del grad_g_vector
        _check_finite("delta_adv", delta_adv)

        torch.nn.utils.vector_to_parameters(w_orig_vector + delta_adv, model_nn_module.parameters())
        del delta_adv

        self.base_optimizer.zero_grad()
        with torch.enable_grad():
            with autocast(device_type="cuda"):
                loss_s, _ = closure_main_loss()

            scaler.scale(loss_s).backward()

        _check_finite("loss_s", loss_s)
        torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())

        scaler.step(self.base_optimizer)

        _enable_running_stats(model_nn_module)

        return loss_original.detach(), loss_s.detach()


class IAM_S_ssl(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float, noise_scale: float, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if noise_scale < 0.0:
            raise ValueError(f"Invalid noise_scale, should be non-negative: {noise_scale}")

        defaults = dict(rho=rho, noise_scale=noise_scale)
        super().__init__(params, defaults)

        self.k_val = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.k_val += p.numel()

        if self.k_val == 0:
            raise ValueError("Optimizer initialized with no trainable parameters.")

        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        if self.param_groups and self.param_groups[0]["params"]:
            self.device = self.param_groups[0]["params"][0].device
        else:
            self.device = torch.device("cpu")
            if self.k_val > 0:
                print(
                    "Warning: Optimizer has k_val > 0 but could not determine device from param_groups. "
                    "Defaulting to CPU."
                )

        self.criterion_kl = nn.KLDivLoss(reduction="batchmean", log_target=False).to(self.device)

    @torch.no_grad()
    def step(self, closure_main_loss, model_nn_module, inputs_for_model):
        if closure_main_loss is None:
            raise ValueError("closure_main_loss is required for this optimizer's step.")

        self.base_optimizer.zero_grad()

        with torch.enable_grad():
            loss_original, outputs_original = closure_main_loss()

        dl_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    dl_list.append(p.grad.data.clone().flatten())

        if not dl_list:
            print("Warning: No gradients (dl) found after initial closure. Skipping optimizer step.")
            return loss_original.detach() if loss_original is not None else None, None

        dl_vector = torch.cat(dl_list).detach()

        eps_val = torch.finfo(outputs_original.dtype).tiny if outputs_original.dtype.is_floating_point else 1e-8
        p_ref = F.softmax(outputs_original, dim=1).clamp_min(eps_val).detach()

        w_orig_vector = torch.nn.utils.parameters_to_vector(model_nn_module.parameters()).detach().clone()
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()

        _disable_running_stats(model_nn_module)

        d_noise = (
            self.defaults["noise_scale"]
            / math.sqrt(self.k_val + 1e-12)
            * torch.randn_like(w_orig_vector, device=self.device)
        )

        torch.nn.utils.vector_to_parameters(w_orig_vector + d_noise, model_nn_module.parameters())

        grad_g_vector = None
        with torch.enable_grad():
            logit_noise, projection_noise = model_nn_module(inputs_for_model)

            active_params_for_g = [p for p in model_nn_module.parameters() if p.requires_grad]
            if not active_params_for_g:
                _enable_running_stats(model_nn_module)
                print(
                    "Warning: No active parameters for 'g' calculation. Restoring original weights and skipping "
                    "perturbation."
                )
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                self.base_optimizer.step()
                return loss_original.detach(), None

            loss_noise_val = self.criterion_kl(F.log_softmax(projection_noise, dim=1), p_ref)

            if torch.isnan(loss_noise_val):
                _enable_running_stats(model_nn_module)
                print("Warning: NaN in loss_noise_val. Restoring original weights and skipping perturbation.")
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                self.base_optimizer.step()
                return loss_original.detach(), None

            grad_g_list = torch.autograd.grad(loss_noise_val, active_params_for_g, create_graph=False)

            flat_grad_g_list = [g.flatten() for g in grad_g_list if g is not None]
            if not flat_grad_g_list:
                _enable_running_stats(model_nn_module)
                print("Warning: Gradient 'g' is empty. Restoring original weights and using original gradients.")
                torch.nn.utils.vector_to_parameters(w_orig_vector, model_nn_module.parameters())
                self.base_optimizer.step()
                return loss_original.detach(), None
            grad_g_vector = torch.cat(flat_grad_g_list)

        delta_adv = self.defaults["rho"] * grad_g_vector / (grad_g_vector.norm(p=2) + 1e-12)

        if dl_vector.numel() == delta_adv.numel():
            delta_adv = torch.sign(torch.dot(delta_adv, dl_vector)) * delta_adv
        else:
            print(
                f"Warning: dl_vector size ({dl_vector.numel()}) and delta_adv (from g) size "
                f"({delta_adv.numel()}) mismatch. Skipping sign alignment for delta."
            )

        torch.nn.utils.vector_to_parameters(w_orig_vector + delta_adv, model_nn_module.parameters())

        self.base_optimizer.zero_grad()
        with torch.enable_grad():
            loss_s, _ = closure_main_loss()

        torch.nn.utils.clip_grad_norm_(model_nn_module.parameters(), max_norm=0.5)

        if torch.isnan(loss_s):
            print("NaN detected in Loss_S at step. Optimizer will use these NaN gradients from Loss_S.")

        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        _enable_running_stats(model_nn_module)

        return loss_original.detach(), loss_s.detach()
