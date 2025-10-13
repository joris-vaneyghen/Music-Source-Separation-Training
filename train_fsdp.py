import argparse
from tqdm.auto import tqdm
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ml_collections import ConfigDict
from typing import List, Callable
import loralib as lora

# Import FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
import torch.distributed as dist

from utils.audio_utils import prepare_data
from utils.settings import parse_args_train, initialize_environment, wandb_init, get_model_from_config
from utils.model_utils import bind_lora_to_model, load_start_checkpoint, save_weights, normalize_batch, \
    initialize_model_and_device, get_optimizer, save_last_weights, save_last_optimizer_state, load_optimizer_state

from utils.losses import choice_loss
from valid import valid_multi_gpu, valid

import warnings

warnings.filterwarnings("ignore")


def setup_fsdp(model: torch.nn.Module, config: ConfigDict) -> FSDP:
    """
    Set up FSDP for the model.

    Args:
        model: The model to wrap with FSDP
        config: Configuration object containing training parameters

    Returns:
        FSDP-wrapped model
    """
    # Configure mixed precision
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=None,  # You can customize this based on your model architecture
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # Important for optimizer compatibility
    )

    return model


def train_one_epoch(model: torch.nn.Module, config: ConfigDict, args: argparse.Namespace,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, device_ids: List[int], epoch: int, use_amp: bool,
                    scaler: torch.cuda.amp.GradScaler,
                    gradient_accumulation_steps: int, train_loader: torch.utils.data.DataLoader,
                    multi_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
    """
    Train the model for one epoch.

    Args:
        model: The model to train (FSDP wrapped).
        config: Configuration object containing training parameters.
        args: Command-line arguments with specific settings (e.g., model type).
        optimizer: Optimizer used for training.
        device: Device to run the model on (CPU or GPU).
        device_ids: List of GPU device IDs if using multiple GPUs.
        epoch: The current epoch number.
        use_amp: Whether to use automatic mixed precision (AMP) for training.
        scaler: Scaler for AMP to manage gradient scaling.
        gradient_accumulation_steps: Number of gradient accumulation steps before updating the optimizer.
        train_loader: DataLoader for the training dataset.
        multi_loss: The loss function to use during training.

    Returns:
        None
    """

    model.train()
    if dist.get_rank() == 0:
        print(f'Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]["lr"]}')

    loss_val = 0.
    total = 0

    normalize = getattr(config.training, 'normalize', False)

    # Create progress bar only on rank 0
    if dist.get_rank() == 0:
        pbar = tqdm(train_loader)
    else:
        pbar = train_loader

    for i, (batch, mixes) in enumerate(pbar):
        x = mixes.to(device)  # mixture
        y = batch.to(device)

        if normalize:
            x, y = normalize_batch(x, y)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if 'roformer' in args.model_type:
                # loss is computed in forward pass
                loss = model(x, y)
            else:
                y_ = model(x)
                loss = multi_loss(y_, y, x)

        loss /= gradient_accumulation_steps
        scaler.scale(loss).backward()

        if config.training.grad_clip:
            # For FSDP, we need to use special gradient clipping
            scaler.unscale_(optimizer)
            model.clip_grad_norm_(config.training.grad_clip)

        if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        li = loss.item() * gradient_accumulation_steps
        loss_val += li
        total += 1

        # Only log from rank 0
        if dist.get_rank() == 0:
            pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
            wandb.log({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'i': i})

        loss.detach()

    if dist.get_rank() == 0:
        print(f'Training loss: {loss_val / total}')
        wandb.log({'train_loss': loss_val / total, 'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})


def compute_epoch_metrics(model: torch.nn.Module, args: argparse.Namespace, config: ConfigDict,
                          device: torch.device, device_ids: List[int], best_metric: float,
                          epoch: int, scheduler: torch.optim.lr_scheduler._LRScheduler) -> float:
    """
    Compute and log the metrics for the current epoch, and save model weights if the metric improves.

    Args:
        model: The model to evaluate (FSDP wrapped).
        args: Command-line arguments containing configuration paths and other settings.
        config: Configuration dictionary containing training settings.
        device: The device (CPU or GPU) used for evaluation.
        device_ids: List of GPU device IDs when using multiple GPUs.
        best_metric: The best metric value seen so far.
        epoch: The current epoch number.
        scheduler: The learning rate scheduler to adjust the learning rate.

    Returns:
        The updated best_metric.
    """

    # For validation, we gather metrics across all processes
    if torch.cuda.is_available() and len(device_ids) > 1:
        metrics_avg, all_metrics = valid_multi_gpu(model, args, config, args.device_ids, verbose=False)
    else:
        metrics_avg, all_metrics = valid(model, args, config, device, verbose=False)

    metric_avg = metrics_avg[args.metric_for_scheduler]

    # Only save on rank 0
    if dist.get_rank() == 0 and metric_avg > best_metric:
        # Save FSDP model
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()

        store_path = f'{args.results_path}/model_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_avg:.4f}.ckpt'
        print(f'Store weights: {store_path}')
        train_lora = args.train_lora
        save_weights(store_path, model_state, device_ids, train_lora)
        best_metric = metric_avg

    # All processes need to step the scheduler
    scheduler.step(metric_avg)

    if dist.get_rank() == 0:
        wandb.log({'metric_main': metric_avg, 'best_metric': best_metric})
        for metric_name in metrics_avg:
            wandb.log({f'metric_{metric_name}': metrics_avg[metric_name]})

    return best_metric


def train_model(args: argparse.Namespace) -> None:
    """
    Trains the model based on the provided arguments, including data preparation, optimizer setup,
    and loss calculation. The model is trained for multiple epochs with logging via wandb.

    Args:
        args: Command-line arguments containing configuration paths, hyperparameters, and other settings.

    Returns:
        None
    """

    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    args = parse_args_train(args)

    # Only initialize environment and wandb on rank 0
    if dist.get_rank() == 0:
        initialize_environment(args.seed, args.results_path)

    model, config = get_model_from_config(args.model_type, args.config_path)
    use_amp = getattr(config.training, 'use_amp', True)

    # For FSDP, we use all available GPUs
    device_ids = list(range(torch.cuda.device_count()))
    batch_size = config.training.batch_size * len(device_ids)

    if dist.get_rank() == 0:
        wandb_init(args, config, device_ids, batch_size)

    train_loader = prepare_data(config, args, batch_size)

    if args.start_check_point:
        load_start_checkpoint(args, model, type_='train')

    if args.train_lora:
        model = bind_lora_to_model(config, model)
        lora.mark_only_lora_as_trainable(model)

    # Wrap model with FSDP instead of regular DataParallel
    model = setup_fsdp(model, config)
    model.to(device)

    if args.pre_valid and dist.get_rank() == 0:
        if torch.cuda.is_available() and len(device_ids) > 1:
            valid_multi_gpu(model, args, config, args.device_ids, verbose=True)
        else:
            valid(model, args, config, device, verbose=True)

        if args.empty_cache:
            torch.cuda.empty_cache()

    optimizer = get_optimizer(config, model)
    gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))

    # Reduce LR if no metric improvements for several epochs
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.patience,
                                  factor=config.training.reduce_factor)

    multi_loss = choice_loss(args, config)
    scaler = GradScaler()
    best_metric = float('-inf')

    if args.optimizer_state:
        epoch, best_metric = load_optimizer_state(args, optimizer, scheduler, device)

    if dist.get_rank() == 0:
        print(
            f"Instruments: {config.training.instruments}\n"
            f"Losses for training: {args.loss}\n"
            f"Metrics for training: {args.metrics}. Metric for scheduler: {args.metric_for_scheduler}\n"
            f"Patience: {config.training.patience} "
            f"Reduce factor: {config.training.reduce_factor}\n"
            f"Batch size: {batch_size} "
            f"Grad accum steps: {gradient_accumulation_steps} "
            f"Effective batch size: {batch_size * gradient_accumulation_steps}\n"
            f"Dataset type: {args.dataset_type}\n"
            f"Optimizer: {config.training.optimizer}"
        )
        print(f'Train for: {config.training.num_epochs} epochs')

    for epoch in range(config.training.num_epochs):
        train_one_epoch(model, config, args, optimizer, device, device_ids, epoch,
                        use_amp, scaler, gradient_accumulation_steps, train_loader, multi_loss)

        if dist.get_rank() == 0:
            save_last_weights(args, model, device_ids)

        if args.empty_cache:
            torch.cuda.empty_cache()

        best_metric = compute_epoch_metrics(model, args, config, device, device_ids, best_metric, epoch, scheduler)

        if dist.get_rank() == 0:
            save_last_optimizer_state(args, epoch, optimizer, scheduler, best_metric)

        if args.empty_cache:
            torch.cuda.empty_cache()

    # Cleanup distributed training
    dist.destroy_process_group()


if __name__ == "__main__":
    import os

    train_model(None)