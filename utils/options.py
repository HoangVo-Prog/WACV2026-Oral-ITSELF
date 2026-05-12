import argparse


_ACTIVE_PROTOTYPE_CONFIG_KEYS = {
    "prototype",
    "use_loss_id",
    "prototype_feature",
    "prototype_per_id",
    "prototype_dim",
    "prototype_kmeans_iters",
    "prototype_warmup_epochs",
    "prototype_tau",
    "prototype_hard_k",
    "prototype_id_weight",
    "prototype_momentum",
    "prototype_lr",
    "prototype_isolated_init",
    "prototype_init_seed",
    "prototype_init_batch_size",
    "prototype_init_num_workers",
}


def sanitize_prototype_args(args, warn_fn=None):
    """Remove stale prototype rank/retrieval fields from loaded configs."""
    removed_fields = []
    for attr in list(vars(args)):
        is_removed_loss_field = attr.startswith("use_loss_") and attr not in _ACTIVE_PROTOTYPE_CONFIG_KEYS
        is_removed_prototype_field = attr.startswith("prototype_") and attr not in _ACTIVE_PROTOTYPE_CONFIG_KEYS
        if not is_removed_loss_field and not is_removed_prototype_field:
            continue
        removed_fields.append(attr)
        delattr(args, attr)

    if removed_fields and warn_fn is not None:
        warn_fn(
            "Removing stale prototype rank/retrieval config fields: {}. "
            "Current CLI supports the training-only prototype ID regularizer.".format(
                ", ".join(sorted(removed_fields))
            )
        )
    return removed_fields


def get_args():
    parser = argparse.ArgumentParser(description="ITSELF Args")
    parser.add_argument("--tau", default=0.015, type=float)
    parser.add_argument("--select_ratio", default=0.4, type=float)
    parser.add_argument("--margin", default=0.1, type=float)
    parser.add_argument("--lambda1_weight", default=0.5, type=float)
    parser.add_argument("--lambda2_weight", default=3.5, type=float)

    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--output_dir", default="run_logs")
    parser.add_argument("--name", default="ITSELF", help="experiment name to save")
    parser.add_argument("--run_time", default="",
                        help="timestamp used in run output paths; defaults to current time")
    parser.add_argument("--log_period", default=20)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
    parser.add_argument("--finetune", type=str, default="")
    parser.add_argument("--finetune_clip", type=str, default="",
                        help="load weights from a CLIP-only ITSELF checkpoint into the current model")
    parser.add_argument("--pretrain", type=str, default="") # unused
    parser.add_argument("--nohup", default=False, action='store_true',
                        help="redirect stdout/stderr to a timestamped .log file")
    parser.add_argument("--nohup_log_dir", default="logs",
                        help="directory for --nohup stdout/stderr logs")


    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=True, action='store_true')
    parser.add_argument("--txt_aug", default=True, action='store_true')
    
    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='tal+cid', help="which loss to use ['cid, tal']")

    ######################## prototype settings ########################
    parser.add_argument(
        "--prototype",
        default=False,
        action='store_true',
        help="enable the training-only prototype regularizer branch",
    )
    parser.add_argument("--use_loss_id", default=False, action='store_true')
    parser.add_argument("--prototype_feature", type=str, default="auto", choices=["auto", "local", "global"])
    parser.add_argument("--prototype_per_id", type=int, default=2)
    parser.add_argument("--prototype_dim", type=int, default=512)
    parser.add_argument("--prototype_kmeans_iters", type=int, default=20)
    parser.add_argument("--prototype_warmup_epochs", type=int, default=0)
    parser.add_argument("--prototype_tau", type=float, default=0.05)
    parser.add_argument("--prototype_hard_k", type=int, default=16,
                        help="top hard negative prototypes used by the prototype ID regularizer")
    parser.add_argument("--prototype_id_weight", type=float, default=0.2)
    parser.add_argument("--prototype_momentum", type=float, default=0.2)
    parser.add_argument(
        "--prototype_isolated_init",
        default=False,
        action='store_true',
        help="initialize prototypes with an isolated deterministic no-augmentation loader",
    )
    parser.add_argument("--prototype_init_seed", type=int, default=1)
    parser.add_argument("--prototype_init_batch_size", type=int, default=None)
    parser.add_argument("--prototype_init_num_workers", type=int, default=None)
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--prototype_lr", type=float, default=None,
                        help="absolute learning rate for prototype_branch; defaults to lr * lr_factor")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(45, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="identity", help="choose sampler from [identity, random]")
    parser.add_argument("--num_instance", type=int, default=2)
    parser.add_argument("--root_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    ### GRAB
    parser.add_argument("--only_global", action='store_true')
    parser.add_argument("--return_all", action='store_true')
    parser.add_argument("--topk_type", type=str, default='mean', help='[mean, std, custom, layer_index]')
    parser.add_argument("--layer_index", type=int, default=-1, help='which layer attention to use: [0, 11]')
    parser.add_argument("--average_attn_weights", type=bool, default=True)
    parser.add_argument("--modify_k", action='store_true')
    
    args = parser.parse_args()
    return args
