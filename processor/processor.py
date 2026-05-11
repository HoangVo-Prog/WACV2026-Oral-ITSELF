import logging
import os
import random
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
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
        or getattr(args, "use_loss_rank", False)
    )


@contextmanager
def _preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def maybe_initialize_prototypes(model, train_loader, args, device, logger):
    model_without_ddp = _unwrap_model(model)
    branch = getattr(model_without_ddp, "prototype_branch", None)
    if branch is None or branch.is_ready():
        return

    logger.info("Initializing identity-aware PBT prototypes from train embeddings")
    logger.info(
        "Prototype ablation settings: kmeans_init={}, group_mean_impl={}, advance_rng_state={}".format(
            getattr(args, "prototype_kmeans_init", "deterministic"),
            getattr(args, "prototype_group_mean_impl", "deterministic"),
            getattr(args, "prototype_advance_rng_state", False),
        )
    )
    was_training = model_without_ddp.training

    dataset = getattr(train_loader, "dataset", None)
    old_txt_aug = getattr(dataset, "txt_aug", None)
    image_features, text_features, pids = [], [], []

    try:
        model_without_ddp.eval()
        if old_txt_aug is not None:
            dataset.txt_aug = False

        rng_context = (
            nullcontext()
            if getattr(args, "prototype_advance_rng_state", False)
            else _preserve_rng_state()
        )
        with rng_context:
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                image_feat, text_feat = model_without_ddp.extract_prototype_features(batch)
                image_feat, text_feat = branch.project_for_memory(image_feat, text_feat)
                image_features.append(image_feat.cpu())
                text_features.append(text_feat.cpu())
                pids.append(batch['pids'].cpu())
    finally:
        if old_txt_aug is not None:
            dataset.txt_aug = old_txt_aug
        model_without_ddp.train(was_training)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    pids = torch.cat(pids, dim=0)
    branch.initialize_projected(image_features, text_features, pids)
    logger.info("Prototype banks initialized with {} samples".format(pids.numel()))


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


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("ITSELF.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    evaluator.eval(model.eval())
    # train
    now_top1 = 0
    current_epoch = 0
    current_steps = 0 
    for epoch in range(start_epoch, num_epoch + 1):
        current_epoch += 1
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

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
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()
            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.count > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {args.lr:.2e}"
                logger.info(info_str)

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
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())
                now_top1 = max(now_top1,top1)
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
