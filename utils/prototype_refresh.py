def validate_prototype_refresh_args(args):
    start_epoch = int(getattr(args, "prototype_refresh_start_epoch", -1))
    step = int(getattr(args, "prototype_refresh_step", -1))
    alpha = float(getattr(args, "prototype_refresh_alpha", 0.35))

    if step == 0:
        raise ValueError("--prototype_refresh_step must be -1 or a positive integer")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("--prototype_refresh_alpha must be in [0, 1]")
    return start_epoch, step, alpha


def prototype_refresh_due(args, epoch):
    start_epoch, step, _ = validate_prototype_refresh_args(args)
    epoch = int(epoch)
    if start_epoch < 0 or epoch < start_epoch:
        return False
    if step < 0:
        return epoch == start_epoch
    return (epoch - start_epoch) % step == 0
