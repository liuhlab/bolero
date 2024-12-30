# %%
from bolero import init
from bolero.tl.model.borzoi_human.train import BorzoiCorigamiHumanLoRATrainer
import torch
init(num_cpus=16, verbose=True)

# %%
name = "2024-12-11-borzoi-corigami-all-cell-type-true-atac"
fold_id = 0
wandb_project = "borzoi_corigami_human_all_cell_types"

# borzoi model
emb_input_features = 50
kv_bottleneck = None

# training config
lr = 0.0002
train_batches = 5000
val_batches = 1000

# cell_types = ["ASC"]
cell_types = ["Amy", "ASC", "CHD7", "EC", "Foxp2", "L4_IT", "L5_ET", "L5_IT", "L6_CT", "L6_IT_Car3", "L6_IT", "L6b", "L23_IT", "L56_NP", "Lamp5_LHX6", "Lamp5", "MGC", "MSN_D1", "MSN_D2", "ODC", "OPC", "PC", "Pvalb_ChC", "Pvalb", "Sncg", "Sst", "SubCtx", "Vip", "VLMC"]
dataset_path = '/large_storage/zhoulab/project/seqmodel/data/bolero_data_path.json'
embeddings_path = '/large_storage/zhoulab/project/seqmodel/data/cell_type_embeddings.json'
base_checkpoint_path = f"/large_storage/zhoulab/hanliu/240729-WMBRNAModel/05.borzoi/borzoi_pretrain/torch_checkpoints/borzoi.human.f{fold_id}.pt"
borzoi_checkpoint_path = "/large_storage/zhoulab/tlgallent/model/241209_atac_all_conditional_scaled.lora.best_checkpoint.pt"
# borzoi_checkpoint_path = "/home/tlgallent/projects/finetune_borzoi/model/20241121_1131_original_settings_all_cell_types_all_all_conditional_lora_ATAC_yishuang_code_replication.lora.best_checkpoint.pt"
# borzoi_checkpoint_path = "/large_storage/zhoulab/yishuang/project/bolero/tests/borzoi/model/test-borzoi-trainer-with-online-loader.lora.best_checkpoint.pt"

config = {
    # file path and wandb
    "output_dir": "model",
    "savename": name,
    "wandb_project": "borzoi_corigami_human_all_cell_types",
    "wandb_job_type": "lora",
    "wandb_name": f"{name}-fold-{fold_id}",
    # train
    "mode": 'base',
    "fold_split_id": fold_id,
    "max_epochs": 200,
    "patience": 10,
    "use_amp": True,
    "use_ema": False,
    "train_batches": train_batches,
    "val_batches": val_batches,
    "batch_size": 4,
    "use_predicted_atac": False,
    "use_dna_embedding": True,
    "lr": lr,
    "weight_decay": 0,
    "std": 0.1,
    "dataloader_concurrency": 6,
    "borzoi_checkpoint_path": borzoi_checkpoint_path,
    # dataset
    "cell_types": cell_types,
    "dataset_path": dataset_path,
    "embeddings_path": embeddings_path,
    "pos_resolution": 32,
    "hic_resolution": 10000,
    "dna_window": 524288,
    "reverse_complement": True,
    "max_jitter": 0,
    # Corigami model
    "seq_len": 16384,
    "in_channel": 1920,
    "output_channel": 256,
    "image_scale": 64,
    "dig_pw_mode": None,
    # borzoi Model
    "base_checkpoint_path": base_checkpoint_path,
    "emb_input_features": emb_input_features,
    "kv_bottleneck": kv_bottleneck,
    "output_layer_groups": 1,
    "lora_scale": 0.02,
    "lora_norm": "layer",
}

config = BorzoiCorigamiHumanLoRATrainer.make_config(**config)

# %%
trainer = BorzoiCorigamiHumanLoRATrainer(config)
# %%
trainer.train()
# %%
