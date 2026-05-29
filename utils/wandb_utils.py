import logging
import os
import re
import time
from pathlib import Path

from utils.comm import get_rank


_WANDB = None
_RUN_NAME_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def _strip_quotes(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_dotenv(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.split(" #", 1)[0])
        if key and key not in os.environ:
            os.environ[key] = value


def load_kaggle_wandb_secrets(logger):
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception:
        return

    if os.environ.get("WANDB_API_KEY"):
        return

    client = UserSecretsClient()
    for name in ("WANDB_API_KEY", "wandb_api_key", "WANDB_KEY", "wandb"):
        try:
            value = client.get_secret(name)
        except Exception:
            continue
        if value:
            os.environ["WANDB_API_KEY"] = value
            return


def _timestamp_run_name():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _resolve_run_name(args, run_name, logger):
    name = getattr(args, "wandb_name", "") or run_name
    if _RUN_NAME_PATTERN.match(name):
        return name
    fallback = _timestamp_run_name()
    logger.warning("WandB run name %s does not match YYYYMMDD_HHMMSS; using %s", name, fallback)
    return fallback


def setup_wandb(args, run_name, logger=None):
    global _WANDB
    logger = logger or logging.getLogger("ITSELF.wandb")

    if get_rank() != 0 or not getattr(args, "wandb", False):
        return None

    load_dotenv()
    load_kaggle_wandb_secrets(logger)

    try:
        import wandb
    except Exception as exc:
        logger.warning("WandB requested but import failed: %s", exc)
        return None

    mode = getattr(args, "wandb_mode", "online") or "online"
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        logger.warning("WandB requested but WANDB_API_KEY was not found in environment, .env, or Kaggle secrets")
        return None

    if os.environ.get("WANDB_API_KEY") and mode != "disabled":
        try:
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
        except Exception as exc:
            logger.warning("WandB login failed: %s", exc)
            return None

    project = getattr(args, "wandb_project", "ITSELF") or "ITSELF"
    entity = getattr(args, "wandb_entity", "") or None
    name = _resolve_run_name(args, run_name, logger)
    tags = getattr(args, "wandb_tags", None) or None

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        mode=mode,
        dir=getattr(args, "output_dir", None),
        config=vars(args),
    )
    _WANDB = wandb
    logger.info("WandB run initialized: project=%s name=%s mode=%s", project, name, mode)
    return run


def wandb_log(metrics, step=None):
    if _WANDB is None or not metrics:
        return
    clean = {key: value for key, value in metrics.items() if value is not None}
    if clean:
        _WANDB.log(clean, step=step)


def _safe_artifact_name(name):
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return clean or "best-checkpoints"


def wandb_upload_best_checkpoints(output_dir, logger=None, metadata=None):
    logger = logger or logging.getLogger("ITSELF.wandb")
    if get_rank() != 0 or _WANDB is None:
        return []

    run = getattr(_WANDB, "run", None)
    if run is None:
        logger.warning("WandB best checkpoint upload skipped because no active run exists")
        return []

    output_path = Path(output_dir)
    checkpoint_paths = sorted(path for path in output_path.glob("best*.pth") if path.is_file())
    if not checkpoint_paths:
        logger.warning("WandB best checkpoint upload skipped; no best*.pth files found in %s", output_path)
        return []

    run_name = getattr(run, "name", None) or getattr(run, "id", None) or "run"
    artifact_name = _safe_artifact_name(f"{run_name}-best-checkpoints")
    artifact_metadata = {
        "output_dir": str(output_path),
        "files": {path.name: path.stat().st_size for path in checkpoint_paths},
    }
    if metadata:
        artifact_metadata.update(metadata)

    artifact = _WANDB.Artifact(
        name=artifact_name,
        type="model",
        metadata=artifact_metadata,
    )
    for path in checkpoint_paths:
        artifact.add_file(str(path), name=path.name)

    try:
        run.log_artifact(artifact, aliases=["best", "latest"])
    except Exception as exc:
        logger.warning("WandB best checkpoint upload failed: %s", exc)
        return []

    logger.info(
        "Uploaded best checkpoint file(s) to WandB artifact %s: %s",
        artifact_name,
        ", ".join(path.name for path in checkpoint_paths),
    )
    return checkpoint_paths


def wandb_finish():
    if _WANDB is not None:
        _WANDB.finish()
