# %%
from bolero import init
from bolero.tl.model.borzoi_human.train import BorzoiCorigamiHumanLoRATrainer
import torch
init(num_cpus=16, verbose=True)

# %%
name = "freeze_all_borzoi_parameters_real_atac"
fold_id = 0

# borzoi model
emb_input_features = 50
kv_bottleneck = None

# training config
lr = 0.0002
train_batches = 1000
val_batches = 50

cell_types = ["ASC"]
dataset_path = '/large_storage/zhoulab/project/seqmodel/data/bolero_data_path.json'
embeddings_path = '/large_storage/zhoulab/project/seqmodel/data/cell_type_embeddings.json'
base_checkpoint_path = f"/large_storage/zhoulab/hanliu/240729-WMBRNAModel/05.borzoi/borzoi_pretrain/torch_checkpoints/borzoi.human.f{fold_id}.pt"
borzoi_checkpoint_path = "/large_storage/zhoulab/yishuang/project/bolero/tests/borzoi/model/test-borzoi-trainer-with-online-loader.lora.best_checkpoint.pt"

config = {
    # file path and wandb
    "output_dir": "model",
    "savename": name,
    "wandb_project": "borzoi_corigami_human",
    "wandb_job_type": "lora",
    "wandb_name": f"{name}-fold-{fold_id}",
    # train
    "max_epochs": 30,
    "patience": 10,
    "use_amp": True,
    "use_ema": False,
    "train_batches": train_batches,
    "val_batches": val_batches,
    "batch_size": 4,
    "use_predicted_atac": False,
    # "accumulate_grad": 64,
    "lr": lr,
    "weight_decay": 0,
    "std": 0.1,
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
    "base_checkpoint_path": base_checkpoint_path,
    "borzoi_checkpoint_path": borzoi_checkpoint_path,
    "emb_input_features": emb_input_features,
    "kv_bottleneck": kv_bottleneck,
}

config = BorzoiCorigamiHumanLoRATrainer.make_config(**config)

# %%
trainer = BorzoiCorigamiHumanLoRATrainer(config)

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
# data_key = "hic"
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
