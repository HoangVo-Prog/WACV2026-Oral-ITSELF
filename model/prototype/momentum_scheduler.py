from bisect import bisect_right
from math import cos, pi


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class PrototypeMomentumScheduler:
    def __init__(
        self,
        mode,
        base_momentum=0.2,
        min_momentum=0.05,
        total_epochs=60,
        warmup_epochs=0,
        milestones=None,
        gamma=0.5,
    ):
        if mode not in ("linear", "cosine", "step"):
            raise ValueError(f"Unsupported prototype momentum scheduler mode: {mode}")
        if total_epochs <= 0:
            raise ValueError("prototype momentum total_epochs must be positive")
        if warmup_epochs < 0:
            raise ValueError("prototype momentum warmup_epochs must be non-negative")
        if base_momentum < 0 or min_momentum < 0:
            raise ValueError("prototype momentum values must be non-negative")
        if min_momentum > base_momentum:
            raise ValueError("prototype_momentum_min must be <= prototype_momentum")

        self.mode = mode
        self.base_momentum = float(base_momentum)
        self.min_momentum = float(min_momentum)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.milestones = sorted(milestones or [])
        self.gamma = float(gamma)
        self.last_momentum = None

    def get_momentum(self, epoch):
        epoch = max(int(epoch), 1)
        elapsed = epoch - 1
        if elapsed < self.warmup_epochs:
            return self.base_momentum

        if self.mode == "step":
            momentum = self.base_momentum * (self.gamma ** bisect_right(self.milestones, epoch))
            return max(self.min_momentum, momentum)

        decay_epochs = max(self.total_epochs - 1 - self.warmup_epochs, 1)
        progress = (elapsed - self.warmup_epochs) / decay_epochs
        progress = min(max(progress, 0.0), 1.0)

        if self.mode == "linear":
            factor = 1.0 - progress
        elif self.mode == "cosine":
            factor = 0.5 * (1.0 + cos(pi * progress))
        else:
            raise NotImplementedError
        return self.min_momentum + (self.base_momentum - self.min_momentum) * factor

    def step(self, model, epoch):
        branch = getattr(_unwrap_model(model), "prototype_branch", None)
        memory = getattr(branch, "memory", None) if branch is not None else None
        if memory is None:
            return None

        momentum = self.get_momentum(epoch)
        memory.momentum = momentum
        self.last_momentum = momentum
        return momentum


def build_prototype_momentum_scheduler(args):
    mode = getattr(args, "prototype_momentum_scheduler", None)
    if mode is None:
        return None

    return PrototypeMomentumScheduler(
        mode=mode,
        base_momentum=getattr(args, "prototype_momentum", 0.2),
        min_momentum=getattr(args, "prototype_momentum_min", 0.05),
        total_epochs=getattr(args, "prototype_momentum_total_epochs", None) or args.num_epoch,
        warmup_epochs=getattr(args, "prototype_momentum_warmup_epochs", 0),
        milestones=getattr(args, "prototype_momentum_milestones", None),
        gamma=getattr(args, "prototype_momentum_gamma", 0.5),
    )
