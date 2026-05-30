# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
from collections import OrderedDict

import torch


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _cpu_state_dict(state_dict):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def _is_model_prototype_memory_key(key):
    if key.startswith("module."):
        key = key[len("module."):]
    return key.startswith("prototype_branch.memory.")


def _is_branch_memory_key(key):
    return key.startswith("memory.")


def _without_model_prototype_memory(state_dict):
    return OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if not _is_model_prototype_memory_key(key)
    )


def _without_branch_memory(state_dict):
    return OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if not _is_branch_memory_key(key)
    )


class Checkpointer:
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = _without_model_prototype_memory(self.model.state_dict())
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def _prototype_branch_and_memory(self):
        model = _unwrap_model(self.model)
        branch = getattr(model, "prototype_branch", None)
        if branch is None:
            self.logger.warning("Skipping prototype branch checkpoint because the model has no prototype branch.")
            return None, None
        return branch, getattr(branch, "memory", None)

    def _prototype_config(self, branch, memory):
        return {
            "feature_dim": getattr(branch, "feature_dim", None),
            "prototype_dim": getattr(branch, "prototype_dim", None),
            "projector_mode": getattr(branch, "projector_mode", None),
            "no_pbt": getattr(getattr(branch, "args", None), "no_pbt", None),
            "use_local": getattr(branch, "use_local", None),
            "num_classes": getattr(memory, "num_classes", None) if memory is not None else None,
            "prototypes_per_id": getattr(memory, "prototypes_per_id", None) if memory is not None else None,
            "dim": getattr(memory, "dim", None) if memory is not None else None,
            "momentum": getattr(memory, "momentum", None) if memory is not None else None,
        }

    def prototype_branch_state(self):
        branch, memory = self._prototype_branch_and_memory()
        if branch is None:
            return None

        data = {
            "prototype_branch": _cpu_state_dict(_without_branch_memory(branch.state_dict())),
            "prototype_ready": bool(memory.is_ready()) if memory is not None and hasattr(memory, "is_ready") else None,
            "prototype_config": self._prototype_config(branch, memory),
        }
        return data

    def prototype_bank_state(self):
        branch, memory = self._prototype_branch_and_memory()
        if branch is None:
            return None
        if memory is not None:
            return {
                "prototype_bank": _cpu_state_dict(memory.state_dict()),
                "prototype_ready": bool(memory.is_ready()) if hasattr(memory, "is_ready") else None,
                "prototype_config": self._prototype_config(branch, memory),
            }
        self.logger.warning("Skipping prototype bank checkpoint because the prototype branch has no memory bank.")
        return None

    def _save_data(self, name, data, description):
        if not self.save_dir:
            return False

        if not self.save_to_disk:
            return False

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving {} to {}".format(description, save_file))
        torch.save(data, save_file)
        return True

    def save_prototype_bank(self, name, **kwargs):
        data = self.prototype_bank_state()
        if data is None:
            return False
        data.update(kwargs)
        return self._save_data(name, data, "prototype bank checkpoint")

    def save_prototype_branch(self, name, **kwargs):
        if not self.save_dir:
            return False

        if not self.save_to_disk:
            return False

        data = self.prototype_branch_state()
        if data is None:
            return False
        data.update(kwargs)

        return self._save_data(name, data, "prototype branch checkpoint")

    def load(self, f=None):
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

    def resume(self, f=None):
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            raise IOError(f"No Checkpoint file found on {f}")
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        # return any further checkpoint data
        return checkpoint

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, except_keys=None):
        load_state_dict(self.model, checkpoint.pop("model"), except_keys)


def check_key(key, except_keys):
    if except_keys is None:
        return False
    else:
        for except_key in except_keys:
            if except_key in key:
                return True
        return False


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys=None):
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger("PersonSearch.checkpoint")
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if check_key(key, except_keys):
            continue
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, except_keys=None):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys)

    # use strict loading
    model.load_state_dict(model_state_dict)
