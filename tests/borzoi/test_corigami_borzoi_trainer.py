# %%
from bolero import init
from bolero.tl.model.borzoi_human.train import BorzoiCorigamiHumanLoRATrainer
import torch
init(num_cpus=16, verbose=True)

# %%
name = "2024-11-28-borzoi-corigami-true-atac"
fold_id = 0
wandb_project = "corigami-all-cell-types"

# borzoi model
emb_input_features = 50
kv_bottleneck = None

# training config
lr = 5e-5
train_batches = 2000
val_batches = 500

cell_types = ["Amy", "ASC", "CHD7", "EC", "Foxp2", "L4_IT", "L5_ET", "L5_IT", "L6_CT", "L6_IT_Car3", "L6_IT", "L6b", "L23_IT", "L56_NP", "Lamp5_LHX6", "Lamp5", "MGC", "MSN_D1", "MSN_D2", "ODC", "OPC", "PC", "Pvalb_ChC", "Pvalb", "Sncg", "Sst", "SubCtx", "Vip", "VLMC"]
dataset_path = '/large_storage/zhoulab/project/seqmodel/data/bolero_data_path.json'
embeddings_path = '/large_storage/zhoulab/project/seqmodel/data/cell_type_embeddings.json'
base_checkpoint_path = f"/large_storage/zhoulab/hanliu/240729-WMBRNAModel/05.borzoi/borzoi_pretrain/torch_checkpoints/borzoi.human.f{fold_id}.pt"
borzoi_checkpoint_path = "/large_storage/zhoulab/hanliu/241123-HumanSequenceModel/human_atac_borzoi/model/241123_atac_baseline.lora.best_checkpoint.pt"
# borzoi_checkpoint_path = "/home/tlgallent/projects/finetune_borzoi/model/20241121_1131_original_settings_all_cell_types_all_all_conditional_lora_ATAC_yishuang_code_replication.lora.best_checkpoint.pt"
# borzoi_checkpoint_path = "/large_storage/zhoulab/yishuang/project/bolero/tests/borzoi/model/test-borzoi-trainer-with-online-loader.lora.best_checkpoint.pt"

config = {
    # file path and wandb
    "output_dir": "model",
    "savename": name,
    "wandb_project": "borzoi_corigami_human",
    "wandb_job_type": "lora",
    "wandb_name": f"{name}-fold-{fold_id}",
    # train
    "mode": 'base',
    "fold_split_id": fold_id,
    "max_epochs": 40,
    "patience": 20,
    "use_amp": True,
    "use_ema": False,
    "train_batches": train_batches,
    "val_batches": val_batches,
    "batch_size": 4,
    "use_predicted_atac": False,
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
    "dig_pw_mode": "concat",
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
# borzoi_model = trainer.borzoi_model_class.create_from_config(trainer.config)
# borzoi_model.convert_to_lora()

# checkpoint = torch.load(trainer.config["borzoi_checkpoint_path"])
# model_weights = checkpoint["state_dict"]
# borzoi_model.load_state_dict(model_weights)

# # %%
# dataset = trainer.dataset
# dataset.train()
# (train_folds, valid_folds, test_folds, train_regions, valid_regions, test_regions) = dataset.get_train_valid_test(fold=0)
# dataloader = dataset.get_dataloader(region_bed=train_regions)

# # %%
# for batch_id, batch in enumerate(dataloader):
#     print(batch_id, batch)
#     break

# # %%
# for key, value in batch.items():
#     print(key, value.shape)
# # %%
# data_key = "hic_E"
# dna_key = "dna_one_hot"
# embedding_key = "cell_type_embedding"

# # ==========
# # Get batch data
# # ==========
# X = batch.get(dna_key, None)
# embedding = batch.get(embedding_key, None)
# # %%
# trainer._setup_model()
# model = trainer.model
# model.to(trainer.device)
# borzoi_model = trainer.borzoi_model
# borzoi_model.to(trainer.device)

# # %%
# atac_count, dna_embedding = borzoi_model.forward(
#     x=X, embedding=embedding, return_dna_embedding=True, crop=False
# )
# atac_log = torch.log1p(atac_count)
# corigami_input = torch.cat([dna_embedding, atac_log], dim=1)
# # %%
# y_true = batch.get(data_key, None)
# y_pred = model(corigami_input)
# %%
trainer.train()
# %%
