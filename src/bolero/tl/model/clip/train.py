import os

import joblib
import torch
import wandb

from bolero.tl.structure.esm import ESMC

from .dataloader import FeatherDataset
from .model import CLIP


def _init_wandb(cfg):
    return wandb.init(
        project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True
    )


def _log_wandb(output, step, wandb_run, index=None):
    metrics_to_log = [
        "loss",
        "auc",
        "top1_acc",
        "top5_acc",
    ]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


def _save_checkpoint(sae, cfg, step):
    save_dir = f"checkpoints/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    ckpt_path = os.path.join(save_dir, "clip.pt")
    torch.save(sae.state_dict(), ckpt_path)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    joblib.dump(cfg, config_path)


def train(model, data_iter, cfg):
    """Train the model."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
    )
    accumulation_steps = cfg["accumulation_steps"]

    wandb_run = _init_wandb(cfg)

    for i, batch in enumerate(data_iter):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model(batch)

        _log_wandb(result, i, wandb_run)

        if (i + 1) % cfg["checkpoint_freq"] == 0:
            _save_checkpoint(model, cfg, i)

        loss = result["loss"] / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()

    _save_checkpoint(model, cfg, i)


def prepare_train(feather_path, name, esmc_model=None, **kwargs):
    """
    Main training function.

    Parameters
    ----------
    feather_path : str
        Path to the feather file containing the dataset.
    name : str
        The name of the wandb run.
    **kwargs
        Optional keyword arguments to override the default configuration of
        BatchTopKSAE model and training.
    """
    default_cfg = {
        "name": name,
        "wandb_project": "clip",
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.98,
        "checkpoint_freq": 100,
        "max_grad_norm": 100,
        "batch_size": 2,
        "accumulation_steps": 1,
        "freeze_encoder": True,
    }
    cfg = {**default_cfg, **kwargs}

    if esmc_model is None:
        esmc = ESMC()
        esmc_model = esmc.model

    model = CLIP(encoder=esmc_model, freeze_encoder=cfg["freeze_encoder"])
    model = model.to("cuda")
    dataset = FeatherDataset(feather_path)
    dataset.train()
    data_iter = dataset.iter_batches(cfg["batch_size"])

    return model, data_iter, cfg
