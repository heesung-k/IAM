"""Factories for optimisers, schedulers, and gradient scalers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.amp import GradScaler

from optim import IAM_S, SAM, IAM_D, IAM_DE


@dataclass
class OptimComponents:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    scaler: Optional[GradScaler]
    scaler: Optional[GradScaler]
    use_amp: bool
    optimizer_name: str
    is_step_scheduler: bool = False


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
    apply_decay_to_all: bool = False,
):
    """
    Split parameters into groups that should have weight decay and those that should not.
    Generally, bias and normalization parameters (1D) are excluded from weight decay.
    When apply_decay_to_all is True, all parameters receive weight decay.
    """
    param_dict = {pn: p for pn, p in model.named_parameters()}
    if apply_decay_to_all:
        return [{"params": [param_dict[pn] for pn in sorted(param_dict)], "weight_decay": weight_decay}]

    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    blacklist_weight_modules = (
        torch.nn.LayerNorm, 
        torch.nn.BatchNorm1d, 
        torch.nn.BatchNorm2d, 
        torch.nn.BatchNorm3d, 
        torch.nn.GroupNorm,
        torch.nn.SyncBatchNorm,
    )

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn  # full param name

            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of linear/conv layers will be decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of normalization layers will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("gamma") or pn.endswith("beta"):
                # some custom norm layers use gamma/beta
                no_decay.add(fpn)

    # Special case for ViT/Transformers (pos_embed, cls_token etc)
    # Check for any parameters that weren't caught by the module traversal
    inter_params = decay | no_decay
    union_params = set(param_dict.keys())
    remaining = union_params - inter_params
    
    for pn in remaining:
        # If it's a 1D parameter, it's likely a bias or scale, so no decay
        if param_dict[pn].ndim < 2:
            no_decay.add(pn)
        else:
            decay.add(pn)

    # Validate that all parameters are accounted for
    assert len(inter_params & remaining) == 0
    assert len(union_params - (decay | no_decay)) == 0

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups


def build_optimizer_and_scheduler(args, model, weight_decay: float, steps_per_epoch: int) -> OptimComponents:
    optimizer_name = args.optimizer
    requested_amp = getattr(args, "amp", False)
    use_amp = requested_amp

    model_name = str(args.model).lower()
    is_vit = "vit" in model_name
    is_cnn = model_name.startswith("resnet") or model_name == "wideresnet"
    if is_cnn:
        effective_wd = args.wd if args.wd is not None else weight_decay
        base_optimizer_cls = torch.optim.SGD
        base_kwargs = {
            "lr": args.lr,
            "momentum": 0.9,
            "weight_decay": effective_wd,
        }
    else:
        print("Using AdamW base optimizer for non-CNN model.")
        
        target_lr = args.lr
        if args.lr == 0.2: # Default value from parser
            target_lr = 3e-3
        
        target_min_lr = args.min_lr
        if args.min_lr == 0.0: # Default value
            target_min_lr = 3e-5
            
        # target_wd = 0.3 # ViT default requested
        effective_wd = args.wd if args.wd is not None else 0.3
        
        base_optimizer_cls = torch.optim.AdamW
        base_kwargs = {
            "lr": target_lr,
            "weight_decay": effective_wd,
        }
    

    if optimizer_name == "IAM_S":
        if requested_amp:
            print("AMP is not currently supported with IAM_S; running in full precision instead.")
        use_amp = False
        optimizer_params = get_parameter_groups(
            model, base_kwargs.pop("weight_decay"), apply_decay_to_all=is_cnn
        )
        optimizer = IAM_S(
            optimizer_params,
            base_optimizer_cls,
            rho=args.rho,
            noise_scale=args.noise_scale,
            **base_kwargs,
        )
    elif optimizer_name == "IAM_D":
        use_amp = False
        optimizer_params = get_parameter_groups(
            model, base_kwargs.pop("weight_decay"), apply_decay_to_all=is_cnn
        )
        optimizer = IAM_D(
            optimizer_params,
            base_optimizer_cls,
            rho=args.rho,
            beta=args.beta,
            noise_scale=args.noise_scale,
            k_steps=args.K,
            **base_kwargs,
        )
    elif optimizer_name == "IAM_DE":
        use_amp = False
        optimizer_params = get_parameter_groups(
            model, base_kwargs.pop("weight_decay"), apply_decay_to_all=is_cnn
        )
        optimizer = IAM_DE(
            optimizer_params,
            base_optimizer_cls,
            rho=args.rho,
            beta=args.beta,
            noise_scale=args.noise_scale,
            **base_kwargs,
        )
    elif optimizer_name == "SAM":
        optimizer_params = get_parameter_groups(
            model, base_kwargs.pop("weight_decay"), apply_decay_to_all=is_cnn
        )
        optimizer = SAM(
            optimizer_params,
            base_optimizer_cls,
            rho=args.rho,
            **base_kwargs,
        )
    elif optimizer_name == "ASAM":
        optimizer_params = get_parameter_groups(
            model, base_kwargs.pop("weight_decay"), apply_decay_to_all=is_cnn
        )
        optimizer = SAM(
            optimizer_params,
            base_optimizer_cls,
            rho=args.rho,
            adaptive=True,
            **base_kwargs,
        )
    elif optimizer_name == "SGD":
        optimizer_params = get_parameter_groups(
            model, base_kwargs.pop("weight_decay"), apply_decay_to_all=is_cnn
        )
        optimizer = base_optimizer_cls(optimizer_params, **base_kwargs)
    
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if getattr(args, "local_rank", 0) == 0:
        print(f"--- Optimizer Groups for {optimizer_name} ---")
        for i, g in enumerate(optimizer.param_groups):
            print(f"Group {i}: weight_decay={g['weight_decay']}, parameters={len(g['params'])}")
        print("------------------------------------------")

    # Scheduler selection logic
    scheduler_type = args.scheduler
    if scheduler_type is None:
        if is_vit:
            scheduler_type = "linear"
        elif args.dataset == "imagenet":
            scheduler_type = "cosine"
        else:
            scheduler_type = "multistep"

    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps
    
    # Default warmup for ViT if not specified (legacy behavior)
    if is_vit and warmup_steps == 0:
        warmup_steps = 10000

    main_steps = total_steps - warmup_steps
    if main_steps < 0:
        main_steps = 1

    # Define Main Scheduler
    if scheduler_type == "linear":
        # Linear decay from 1.0 to min_lr_ratio
        target_min_lr = getattr(args, "min_lr", 0.0)
        if target_min_lr == 0.0 and is_vit:
             target_min_lr = 3e-5
        
        base_lr = base_kwargs["lr"]
        ratio = target_min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(current_step: int):
            # Linearly decay from 1.0 to ratio
            progress = float(current_step) / float(max(1, main_steps))
            return max(0.0, (1.0 - progress) * 1.0 + progress * ratio)
            
        main_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_steps, eta_min=getattr(args, "min_lr", 0.0)
        )

    elif scheduler_type == "multistep":
        # Check if milestones are provided? For now, keep legacy hardcoded logic
        milestones_epochs = [args.epochs * k // 10 for k in (3, 6, 8)]
        # We need to adjust milestones because they are relative to the *start of training*,
        # but the main scheduler sees steps starting from 0 (which is actually step=warmup_steps).
        # However, MultiStepLR typically works on absolute epochs/steps.
        # But SequentialLR resets the counter for the second scheduler?
        # No, SequentialLR invokes the second scheduler with `epoch - milestones[0]`.
        # So main_scheduler sees 0 when real step is warmup_steps.
        
        # We need to convert absolute epoch milestones to steps relative to main_scheduler start
        milestones_steps = [m * steps_per_epoch - warmup_steps for m in milestones_epochs]
        milestones_steps = [m for m in milestones_steps if m > 0]
        
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_steps, gamma=0.2)
        
    elif scheduler_type == "constant":
        def lr_lambda(current_step: int):
            return 1.0
        main_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


    # Combine with Warmup
    if warmup_steps > 0:
        # Linear Warmup from start_factor to 1.0
        # Start factor ~ 0.0
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.0001, end_factor=1.0, total_iters=warmup_steps
        )
        # Combine
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
        )
    else:
        scheduler = main_scheduler

    # All schedulers are now step-based
    is_step_scheduler = True

    scaler = GradScaler(enabled=use_amp) if use_amp else None

    return OptimComponents(
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        use_amp=use_amp,
        optimizer_name=optimizer_name,
        is_step_scheduler=is_step_scheduler,
    )
