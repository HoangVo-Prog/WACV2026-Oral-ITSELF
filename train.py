import os
import os.path as op
import subprocess
import sys
import torch
import numpy as np
import random
import time
from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize
import warnings
warnings.filterwarnings("ignore")


NOHUP_CHILD_ENV = "ITSELF_NOHUP_CHILD"


def get_session_name(cur_time, name, loss_names):
    return f'{cur_time}_{name}_{loss_names}'


def get_nohup_log_path(args, cur_time, name, rank=0):
    session_name = get_session_name(cur_time, name, args.loss_names)
    log_dir = op.join(args.nohup_log_dir, args.dataset_name, session_name)
    log_name = f"{cur_time}.log" if rank == 0 else f"{cur_time}_rank{rank}.log"
    return op.join(log_dir, log_name)


def detach_nohup_process(args, cur_time, name):
    if not args.nohup or os.environ.get(NOHUP_CHILD_ENV) == "1":
        return

    log_path = get_nohup_log_path(args, cur_time, name)
    os.makedirs(op.dirname(log_path), exist_ok=True)

    child_cmd = [sys.executable] + sys.argv
    if not args.run_time:
        child_cmd.extend(["--run_time", cur_time])

    child_env = os.environ.copy()
    child_env[NOHUP_CHILD_ENV] = "1"

    with open(log_path, "a", buffering=1) as log_file:
        process = subprocess.Popen(
            child_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            env=child_env,
            close_fds=True,
            start_new_session=True,
        )

    print(f"PID: {process.pid}")
    print(f"Log file: {log_path}")
    raise SystemExit(0)


def enable_nohup_logging(log_dir, cur_time, rank=0):
    os.makedirs(log_dir, exist_ok=True)

    log_name = f"{cur_time}.log" if rank == 0 else f"{cur_time}_rank{rank}.log"
    log_path = op.join(log_dir, log_name)
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    return log_path, log_file


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _strip_module_prefix(key):
    return key[7:] if key.startswith("module.") else key


def _checkpoint_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_clip_finetune(model, checkpoint_path, logger):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loaded_state = _checkpoint_model_state(checkpoint)
    model_state = model.state_dict()
    update_state = {}
    skipped_proto = 0
    skipped_missing = 0
    skipped_shape = 0

    for key, value in loaded_state.items():
        key = _strip_module_prefix(key)
        if key.startswith("prototype_branch."):
            skipped_proto += 1
            continue
        if key not in model_state:
            skipped_missing += 1
            continue
        if model_state[key].shape != value.shape:
            skipped_shape += 1
            continue
        update_state[key] = value.detach().clone()

    if not update_state:
        raise RuntimeError(f"No compatible weights found in --finetune_clip checkpoint: {checkpoint_path}")

    model_state.update(update_state)
    model.load_state_dict(model_state)
    logger.info(
        "Loaded %d tensors from CLIP checkpoint %s; skipped %d prototype, %d missing, %d shape-mismatch tensors",
        len(update_state),
        checkpoint_path,
        skipped_proto,
        skipped_missing,
        skipped_shape,
    )

if __name__ == '__main__':
    args = get_args()
    name = "ITSELF"
    cur_time = args.run_time or time.strftime("%Y%m%d_%H%M%S", time.localtime())

    detach_nohup_process(args, cur_time, name)

    set_seed(1+get_rank())

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    session_name = get_session_name(cur_time, name, args.loss_names)
    args.output_dir = op.join(args.output_dir, args.dataset_name, session_name)
    nohup_log_file = None
    if args.nohup:
        nohup_log_dir = op.join(args.nohup_log_dir, args.dataset_name, session_name)
        nohup_log_path, nohup_log_file = enable_nohup_logging(nohup_log_dir, cur_time, get_rank())
    logger = setup_logger('ITSELF', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    if args.nohup:
        logger.info(f"Nohup log file: {nohup_log_path}")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)
    if not os.path.isdir(args.output_dir+'/img'):
        os.makedirs(args.output_dir+'/img')

        
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    if args.finetune:
        logger.info("loading {} model".format(args.finetune))
        param_dict = torch.load(args.finetune,map_location='cpu')['model']
        for k in list(param_dict.keys()):
            refine_k = k.replace('module.','')
            param_dict[refine_k] = param_dict[k].detach().clone()
            del param_dict[k]
        model.load_state_dict(param_dict, False)
    if args.finetune_clip:
        load_clip_finetune(model, args.finetune_clip, logger)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)


    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader, args)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']
        logger.info(f"===================>start {start_epoch}")

    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
