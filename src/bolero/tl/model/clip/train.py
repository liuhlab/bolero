import os

import joblib
import torch
import wandb
from peft import LoraConfig, get_peft_model

from bolero.tl.structure.esm import ESMC

from .dataloader import FeatherDataset
from .model import CLIP


def _init_wandb(cfg):
    return wandb.init(
        project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True
    )


def _log_wandb(output, wandb_run, prefix="train"):
    metrics_to_log = [
        "loss",
        "auc",
        "top1_acc",
        "top5_acc",
    ]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}
    log_dict = {f"{prefix}/{k}": v for k, v in log_dict.items()}
    wandb_run.log(log_dict)


def _save_checkpoint(sae, cfg, step):
    save_dir = "checkpoints/"
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"{cfg['name']}_"

    # Save model state
    ckpt_path = f"{save_dir}{prefix}clip.pt"
    torch.save(sae.state_dict(), ckpt_path)

    # Save config
    config_path = f"{save_dir}{prefix}config.joblib"
    joblib.dump(cfg, config_path)


def get_esmc_model(lora_rank=64, finetune_type="lora"):
    """Get the ESMC model with LoRA configuration."""
    esmc = ESMC()
    model = esmc.model

    # 2. Define LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,  # Rank of low-rank adapters
        lora_alpha=2 * lora_rank,  # Scaling factor
        lora_dropout=0.0,  # Dropout for LoRA layers
        # Adjust target_modules to match your model's attention/linear layer names
        target_modules=["out_proj", "ffn.1", "ffn.3"],
        bias="none",
        use_dora=finetune_type == "dora",
    )

    # 3. Inject LoRA adapters into the model
    model = get_peft_model(model, lora_config)

    # Print information about trainable parameters
    model.print_trainable_parameters()
    return model


def train(model, dataset, cfg):
    """Train the model."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
    )
    accumulation_steps = cfg["accumulation_steps"]

    wandb_run = _init_wandb(cfg)

    # train
    dataset.train()
    train_data_iter = dataset.iter_batches(cfg["batch_size"], cfg["groupby_go_and_tax"])
    for i, batch in enumerate(train_data_iter):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model(batch)

        _log_wandb(result, wandb_run, "train")

        if (i + 1) % cfg["checkpoint_freq"] == 0:
            _save_checkpoint(model, cfg, i)

        loss = result["loss"] / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()

    _save_checkpoint(model, cfg, i)

    # eval
    dataset.eval()
    eval_data_iter = dataset.iter_batches(
        cfg["eval_batch_size"],
        cfg["groupby_go_and_tax"],
        nbatch=cfg["max_eval_batches"],
    )
    for batch in eval_data_iter:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            result = model(batch)
        _log_wandb(result, wandb_run, "eval")

    wandb_run.finish()
    return


def prepare_train(feather_path, name, esmc_model=None, pdb_go_tax_path=None, **kwargs):
    """
    Main training function.

    Parameters
    ----------
    feather_path : str
        Path to the feather file containing the dataset.
    name : str
        The name of the wandb run.
    **kwargs
        Optional keyword arguments to override the default configuration.
    """
    default_cfg = {
        "name": name,
        "wandb_project": "clip",
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.98,
        "checkpoint_freq": 99999,
        "max_grad_norm": 100,
        "batch_size": 2,
        "accumulation_steps": 1,
        "freeze_encoder": True,
        "train_folds": [0, 1, 2, 3, 4, 5, 6],
        "eval_folds": [7],
        "lora_rank": 64,
        "finetune_type": "lora",
        "groupby_go_and_tax": False,
        "concat_dim": "batch",
        "eval_batch_size": 128,
        "max_eval_batches": 500,
    }
    cfg = {**default_cfg, **kwargs}

    if esmc_model is None:
        esmc_model = get_esmc_model(cfg["lora_rank"], cfg["finetune_type"])

    model = CLIP(
        encoder=esmc_model,
        freeze_encoder=cfg["freeze_encoder"],
        concat_dim=cfg["concat_dim"],
    )
    model = model.to("cuda")
    dataset = FeatherDataset(
        feather_path,
        train_folds=cfg["train_folds"],
        eval_folds=cfg["eval_folds"],
        pdb_go_tax_path=pdb_go_tax_path,
        concat_dim=cfg["concat_dim"],
    )
    return model, dataset, cfg


def from_checkpoint(prefix, merge_lora=True):
    """Create a model from a checkpoint."""
    config_path = f"{prefix}_config.joblib"
    cfg = joblib.load(config_path)

    esmc_model = get_esmc_model(
        cfg["lora_rank"], cfg["finetune_type"]
    )  # Load the ESMC model with the same configuration

    model = CLIP(encoder=esmc_model, freeze_encoder=cfg["freeze_encoder"])
    model.load_state_dict(torch.load(f"{prefix}_clip.pt", weights_only=True))

    if merge_lora and hasattr(model.encoder, "merge_and_unload"):
        print("Merged LoRA weights into the base model.")
        model.encoder = model.encoder.merge_and_unload()

    model = model.to("cuda")
    return model
