import json
import os

import torch
import tqdm
import wandb

from .activation_store import ActivationsStore
from .model import BatchTopKSAE


def _init_wandb(cfg):
    return wandb.init(
        project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True
    )


def _log_wandb(output, step, wandb_run, index=None):
    metrics_to_log = [
        "loss",
        "l2_loss",
        "l1_loss",
        "l0_norm",
        "l1_norm",
        "aux_loss",
        "num_dead_features",
        "ave_batches_not_active",
    ]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}
    log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


def _save_checkpoint(sae, cfg, step):
    save_dir = f"checkpoints/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    # # Create and log artifact
    # artifact = wandb.Artifact(
    #     name=f"{cfg['name']}_{step}",
    #     type="model",
    #     description=f"Model checkpoint at step {step}",
    # )
    # artifact.add_file(sae_path)
    # artifact.add_file(config_path)
    # wandb_run.log_artifact(artifact)

    # print(f"Model and config saved as artifact at step {step}")


def _train_sae(sae, activation_store, cfg):
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(
        sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
    )

    wandb_run = _init_wandb(cfg)

    for i in tqdm.trange(num_batches):
        try:
            batch = activation_store.next_batch().float()
            sae_output = sae(batch)
            if (i + 1) % 10 == 0:
                _log_wandb(sae_output, i, wandb_run)

            if (i + 1) % cfg["checkpoint_freq"] == 0:
                _save_checkpoint(sae, cfg, i)

            loss = sae_output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()
        except StopIteration:
            print(f"All activation fetched at step {i}")
            break

    _save_checkpoint(sae, cfg, i)


def train_sae(activation_iter, name, **kwargs):
    """
    Train a BatchTopKSAE model with the given activation iterator.

    Parameters
    ----------
    activation_iter : Iterator[torch.Tensor]
        An iterator that yields activation tensors of shape (batch_size, act_size).
        batch_size may vary between batches, but act_size should be fixed.
    name : str
        The name of the wandb run.
    **kwargs
        Optional keyword arguments to override the default configuration of
        BatchTopKSAE model and training.
    """
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "lr": 2e-5,
        "num_tokens": int(1e9),
        # BatchTopKSAE do not need L1 regularization, since sparsity is controlled by top-k
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "dtype": torch.float32,
        "act_size": 1920,
        "dict_size": 7680,
        "device": "cuda",
        "name": name,
        "wandb_project": "sparse_autoencoders",
        "input_unit_norm": True,
        "checkpoint_freq": 30000,
        "n_batches_to_dead": 5,
        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1 / 32),
    }
    cfg = {**default_cfg, **kwargs}
    if "num_batches_in_buffer" not in cfg:
        cfg["num_batches_in_buffer"] = 2_000_000 // cfg["batch_size"] + 1

    sae_model = BatchTopKSAE(cfg).to(cfg["device"]).to(cfg["dtype"])
    activation_store = ActivationsStore(
        activation_iter,
        cfg["batch_size"],
        cfg["num_batches_in_buffer"],
    )

    _train_sae(sae_model, activation_store, cfg)
    return
