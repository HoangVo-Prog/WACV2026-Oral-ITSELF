import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from datasets.bases import ImageTextDataset
from datasets.build import build_transforms, collate, make_data_loader_generator, seed_worker
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from utils.rng import preserve_rng_state
from utils.train_diagnostics import compute_train_diagnostics
from utils.wandb_utils import wandb_log
from torch.utils.tensorboard import SummaryWriter


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _prototype_ready(model):
    model = _unwrap_model(model)
    branch = getattr(model, "prototype_branch", None)
    return branch is not None and branch.is_ready()


def _prototype_requested(args):
    return (
        getattr(args, "prototype", False)
        or getattr(args, "use_loss_id", False)
    )


def _set_epoch_on_loader(loader, epoch):
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    batch_sampler = getattr(loader, "batch_sampler", None)
    inner_sampler = getattr(batch_sampler, "sampler", None)
    if inner_sampler is not None and hasattr(inner_sampler, "set_epoch"):
        inner_sampler.set_epoch(epoch)


def _build_prototype_init_loader(train_loader, args):
    train_set = getattr(train_loader, "dataset", None)
    source_dataset = getattr(train_set, "dataset", None)
    if source_dataset is None:
        raise RuntimeError("Prototype full-dataset initialization requires train_loader.dataset.dataset")

    prototype_set = ImageTextDataset(
        source_dataset,
        args,
        transform=build_transforms(img_size=args.img_size, aug=False, is_train=False),
        text_length=getattr(train_set, "text_length", args.text_length),
        truncate=getattr(train_set, "truncate", True),
    )
    prototype_set.txt_aug = False
    prototype_set.img_aug = False

    return DataLoader(
        prototype_set,
        batch_size=getattr(args, "test_batch_size", args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=make_data_loader_generator(args, offset=4000),
    )


def _validate_prototype_init_coverage(pids, num_classes, expected_samples):
    sample_count = pids.numel()
    if sample_count != expected_samples:
        raise RuntimeError(
            "Prototype full-dataset initialization saw {} samples, expected {}".format(
                sample_count,
                expected_samples,
            )
        )

    unique_pids = torch.unique(pids.long().cpu(), sorted=True)
    expected_pids = torch.arange(num_classes, dtype=unique_pids.dtype)
    if unique_pids.numel() == num_classes and torch.equal(unique_pids, expected_pids):
        return

    valid_pids = unique_pids[(unique_pids >= 0) & (unique_pids < num_classes)]
    seen = torch.zeros(num_classes, dtype=torch.bool)
    seen[valid_pids] = True
    missing = seen.logical_not().nonzero(as_tuple=False).flatten().tolist()
    preview = missing[:20]
    suffix = "" if len(missing) <= 20 else "..."
    raise RuntimeError(
        "Prototype full-dataset initialization missing {} identities: {}{}".format(
            len(missing),
            preview,
            suffix,
        )
    )


@torch.no_grad()
def maybe_initialize_prototypes(model, train_loader, args, device, logger):
    model_without_ddp = _unwrap_model(model)
    branch = getattr(model_without_ddp, "prototype_branch", None)
    if branch is None or branch.is_ready():
        return

    logger.info("Initializing identity-aware PBT prototypes from train embeddings")
    was_training = model_without_ddp.training

    prototype_loader = _build_prototype_init_loader(train_loader, args)
    expected_samples = len(prototype_loader.dataset)
    logger.info("Using full train dataset for prototype initialization")

    dataset = getattr(prototype_loader, "dataset", None)
    old_txt_aug = getattr(dataset, "txt_aug", None)
    image_features, text_features, pids = [], [], []
    sample_count = 0

    try:
        model_without_ddp.eval()
        if old_txt_aug is not None:
            dataset.txt_aug = False

        with preserve_rng_state(getattr(args, "prototype_seed", 1001)):
            for batch in prototype_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                image_feat, text_feat = model_without_ddp.extract_prototype_features(batch)
                image_feat, text_feat = branch.project_for_memory(image_feat, text_feat)
                image_features.append(image_feat.cpu())
                text_features.append(text_feat.cpu())
                pids.append(batch['pids'].cpu())

            image_features = torch.cat(image_features, dim=0)
            text_features = torch.cat(text_features, dim=0)
            pids = torch.cat(pids, dim=0)
            sample_count = pids.numel()
            _validate_prototype_init_coverage(pids, branch.memory.num_classes, expected_samples)
            branch.initialize_projected(image_features, text_features, pids)
    finally:
        if old_txt_aug is not None:
            dataset.txt_aug = old_txt_aug
        model_without_ddp.train(was_training)

    logger.info("Prototype banks initialized with {} samples".format(sample_count))


def _loss_components(ret):
    return {
        key: value
        for key, value in ret.items()
        if "loss" in key and torch.is_tensor(value)
    }


def _grad_norm_by_loss(losses, model):
    params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    norms = {}
    if not params:
        return norms

    for name, loss in losses.items():
        if not loss.requires_grad:
            norms[f"{name}_grad_norm"] = 0.0
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        grad_sq_sum = loss.new_zeros(())
        has_grad = False
        for grad in grads:
            if grad is None:
                continue
            has_grad = True
            grad_sq_sum = grad_sq_sum + grad.detach().float().pow(2).sum()
        norms[f"{name}_grad_norm"] = grad_sq_sum.sqrt().item() if has_grad else 0.0
    return norms


def _update_meter(meters, key, value, batch_size):
    if key not in meters:
        meters[key] = AverageMeter()
    if torch.is_tensor(value):
        value = value.detach().item()
    meters[key].update(value, batch_size)


def _best_val_wandb_metrics(best_metrics):
    metrics = {}
    for key in ("R1", "R5", "R10", "mAP", "mINP", "rSum"):
        if key in best_metrics:
            metrics[f"val/best_row/{key}"] = best_metrics[key]
    if best_metrics.get("task"):
        metrics["val/best_row_task"] = best_metrics["task"]
    return metrics


def _train_wandb_metrics(meters, loss_components, optimizer, epoch, current_steps):
    metrics = {
        "train/epoch": epoch,
        "train/iteration": current_steps,
        "train/total_loss": meters["loss"].avg,
        "train/weighted_loss": meters["loss"].avg,
        "train/lr": optimizer.param_groups[0]["lr"],
    }
    lrs = [group["lr"] for group in optimizer.param_groups]
    metrics["train/lr_min"] = min(lrs)
    metrics["train/lr_max"] = max(lrs)

    for loss_key in loss_components.keys():
        if loss_key in meters and meters[loss_key].count > 0:
            metrics[f"train/weighted_loss/{loss_key}"] = meters[loss_key].avg

    for key, meter in meters.items():
        if key.endswith("_grad_norm") and meter.count > 0:
            loss_name = key[: -len("_grad_norm")]
            metrics[f"train/loss_grad_norm/{loss_name}"] = meter.avg

    dashboard_keys = [
        "host_margin_mean",
        "host_margin_p10",
        "hard_pos_margin_mean",
        "negative_intrusion_rate",
        "mean_first_positive_rank",
        "proto_margin_img_mean",
        "proto_margin_txt_mean",
        "negative_proto_margin_rate",
        "dead_slot_rate",
        "effective_slots_per_id",
        "slot_redundancy",
        "assignment_flip_rate",
        "hard_negative_overlap",
        "proto_to_host_margin_corr",
    ]
    for key in dashboard_keys:
        if key in meters and meters[key].count > 0:
            metrics[f"train/{key}"] = meters[key].avg
    return metrics


def _train_console_metrics(meters, loss_components):
    keys = ["loss"]
    for loss_key in loss_components.keys():
        keys.append(loss_key)
        keys.append(f"{loss_key}_grad_norm")
    return keys


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0
    arguments["epoch"] = start_epoch - 1

    logger = logging.getLogger("ITSELF.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    initial_eval = evaluator.eval(model.eval(), return_metrics=(get_rank() == 0))
    if get_rank() == 0 and isinstance(initial_eval, tuple):
        initial_top1, _, initial_best_metrics = initial_eval
        wandb_metrics = {
            "val/epoch": 0,
            "val/top1": initial_top1,
        }
        wandb_metrics.update(_best_val_wandb_metrics(initial_best_metrics))
        wandb_log(wandb_metrics, step=0)
    # train
    now_top1 = 0
    current_epoch = 0
    current_steps = 0
    train_diag_state = {"assignments": {}}
    for epoch in range(start_epoch, num_epoch + 1):
        current_epoch += 1
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        _set_epoch_on_loader(train_loader, epoch)
        if _prototype_requested(args):
            if epoch > getattr(args, "prototype_warmup_epochs", 1) and not _prototype_ready(model):
                maybe_initialize_prototypes(model, train_loader, args, device, logger)

        model.train()
        model.epoch = epoch

        
        for n_iter, batch in enumerate(train_loader):
            current_steps += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            if args.modify_k:
                ret = model(batch, epoch, current_step=current_steps)
            else:
                ret = model(batch, epoch)
            loss_components = _loss_components(ret)
            total_loss = sum(loss_components.values())
            batch_size = batch['images'].shape[0]
            _update_meter(meters, 'loss', total_loss, batch_size)
            for loss_key, loss_value in loss_components.items():
                _update_meter(meters, loss_key, loss_value, batch_size)
            if (n_iter + 1) % log_period == 0:
                grad_norms = _grad_norm_by_loss(loss_components, model)
                for grad_key, grad_norm in grad_norms.items():
                    _update_meter(meters, grad_key, grad_norm, batch_size)
                train_diag_metrics = compute_train_diagnostics(model, ret, args, train_diag_state)
                for diag_key, diag_value in train_diag_metrics.items():
                    _update_meter(meters, diag_key, diag_value, batch_size)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()
            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k in _train_console_metrics(meters, loss_components):
                    v = meters.get(k)
                    if v is not None and v.count > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {args.lr:.2e}"
                logger.info(info_str)
                if get_rank() == 0:
                    wandb_log(_train_wandb_metrics(meters, loss_components, optimizer, epoch, current_steps),
                              step=current_steps)

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.count > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0: 
        # if epoch % eval_period == 0 and epoch >= 61:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1, _, best_val_metrics = evaluator.eval(model.module.eval(), return_metrics=True)
                else:
                    top1, _, best_val_metrics = evaluator.eval(model.eval(), return_metrics=True)
                now_top1 = max(now_top1,top1)
                wandb_metrics = {
                    "val/epoch": epoch,
                    "val/top1": top1,
                    "val/best_top1": max(best_top1, top1),
                }
                wandb_metrics.update(_best_val_wandb_metrics(best_val_metrics))
                wandb_log(wandb_metrics, step=current_steps)
                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
                
 
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

                   
def do_inference(model, test_img_loader, test_txt_loader, args):

    logger = logging.getLogger("ITSELF.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    _ = evaluator.eval(model.eval())
