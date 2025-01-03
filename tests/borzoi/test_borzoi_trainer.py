# %%
from bolero import init
from bolero.tl.model.borzoi_human.train import BorzoiHumanLoRATrainer

init(num_cpus=32, verbose=True)

# %%
name = "test-borzoi-human-trainer-with-online-loader"
fold_id = 0
# Conditional lora
lora_preset = "all_conditional"
lora_scale = 0.02
emb_input_features = 50
# pseudobulk and embedding
kv_bottleneck = None
# base model
context_out = 1
n_cycles = 1
loss_total_weight = 0.2

warmup_steps = 100
lr = 5e-5
train_batches = 2000
val_batches = 500

cell_types = ["ASC", "Vip", "L5_IT"]
dataset_path = '/large_storage/zhoulab/project/seqmodel/data/bolero_data_path.json'
embeddings_path = '/large_storage/zhoulab/project/seqmodel/data/cell_type_embeddings.json'
checkpoint_path = f"/large_storage/zhoulab/hanliu/240729-WMBRNAModel/05.borzoi/borzoi_pretrain/torch_checkpoints/borzoi.human.f{fold_id}.pt"

config = {
    # file path and wandb
    "output_dir": "model",
    "savename": name,
    "wandb_project": "borzoi_finetuning",
    "wandb_job_type": "lora",
    "wandb_name": f"{name}-fold-{fold_id}",
    # train
    "start_early_stop_after_epoch": 15,
    "max_epochs": 30,
    "patience": 10,
    "use_amp": True,
    "use_ema": False,
    "train_batches": train_batches,
    "val_batches": val_batches,
    "batch_size": 2,
    "accumulate_grad": 64,
    "lr": lr,
    "lora_scale": lora_scale,
    "warmup_steps": warmup_steps,
    "weight_decay": 1e-8,
    "global_clipnorm": 0.1,
    # dataset
    "fold_split_id": 0,
    "cell_types": cell_types,
    "dataset_path": dataset_path,
    "embeddings_path": embeddings_path,
    "resolution": 10000,
    "dna_window": 524288,
    "pos_resolution": 32,
    "reverse_complement": True,
    "max_jitter": 3,
    # LoRA Model
    "base_checkpoint_path": checkpoint_path,
    "emb_input_features": emb_input_features,
    "out_channels": 1,
    "hidden_dim": 256,
    "hidden_layers": 1,
    "output_layer_groups": 4,
    "lora_dropout": 0.01,
    "loss_total_weight": loss_total_weight,
    "lora_preset": lora_preset,
    # base model dropout:
    "transformer_attn_dropout": 0.0,
    "transformer_pos_dropout": 0.0,
    "transformer_ff_dropout": 0.0,
    "final_conv_dropout": 0.0,
    "final_output_dropout": 0.01,
    # pseudobulk and embedding
    "kv_bottleneck": kv_bottleneck,
    "num_memories": 256,
    "dim_memory": 20,
    "num_memory_codebooks": 2,
    "additional_embs": 1,
    "n_cycles": n_cycles,
}

config = BorzoiHumanLoRATrainer.make_config(**config)

# %%
trainer = BorzoiHumanLoRATrainer(config)
# %%
trainer.train()
# %%
