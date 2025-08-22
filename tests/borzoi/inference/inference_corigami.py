# %%
import argparse
import pandas as pd
import scanpy as sc

from bolero.tl.model.borzoi_human.infer import BorzoiCorigamiInferencer
from bolero import init

# Set up
init(num_cpus=16, object_store_memory_gb=200, verbose=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="corigami inference")
    parser.add_argument('--cell_type', type=str, help="The cell type to train")
    parser.add_argument('--checkpoint_name', type=str, help="The name of the checkpoint runc")
    parser.add_argument('--inference_folder', type=str, default='corigami', help="The name of inference folder")
    args = parser.parse_args()
    return args

# %%
def main():
    args = parse_arguments()
    cell_types = [args.cell_type]
    fold_id = 0
    # cell_type = ["Amy", "ASC", "CHD7", "EC", "Foxp2", "L4_IT", "L5_ET", "L5_IT", "L6_CT", "L6_IT_Car3", "L6_IT", "L6b", "L23_IT", "L56_NP", "Lamp5_LHX6", "Lamp5", "MGC", "MSN_D1", "MSN_D2", "ODC", "OPC", "PC", "Pvalb_ChC", "Pvalb", "Sncg", "Sst", "SubCtx", "Vip", "VLMC"]
    dataset_path = '/large_storage/zhoulab/project/seqmodel/data/bolero_data_path.json'
    embeddings_path = '/large_storage/zhoulab/project/seqmodel/data/cell_type_embeddings.json'
    base_checkpoint_path = f"/large_storage/zhoulab/hanliu/240729-WMBRNAModel/05.borzoi/borzoi_pretrain/torch_checkpoints/borzoi.human.f{fold_id}.pt"
    borzoi_checkpoint_path = "/large_storage/zhoulab/hanliu/241123-HumanSequenceModel/human_atac_borzoi/model/241123_atac_baseline.lora.best_checkpoint.pt"

    # borzoi model
    emb_input_features = 50
    kv_bottleneck = None

    # training config
    lr = 0.0002
    train_batches = 1000
    val_batches = 500

    inference_folder = args.inference_folder

    config = {
        # file path and wandb
        "output_dir": f"/large_storage/zhoulab/yishuang/project/bolero/tests/borzoi/{inference_folder}/{args.cell_type}",
        "savename": "base",
        "wandb_project": "borzoi_corigami_human",
        "wandb_job_type": "inference",
        "wandb_name": "borzoi_corigami",
        # train
        "mode": 'base',
        "fold_split_id": fold_id,
        "max_epochs": 40,
        "patience": 40,
        "use_amp": True,
        "use_ema": False,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "batch_size": 2,
        "use_predicted_atac": True,
        "use_dna_embedding": True,
        "lr": lr,
        "weight_decay": 0,
        "std": 0.1,
        "dataloader_concurrency": 4,
        "borzoi_checkpoint_path": borzoi_checkpoint_path,
        "pretrained_model": f"/large_storage/zhoulab/yishuang/project/bolero/tests/borzoi/model/{args.checkpoint_name}.base.best_checkpoint.pt",
        # dataset
        "cell_types": cell_types,
        "dataset_path": dataset_path,
        "embeddings_path": embeddings_path,
        "pos_resolution": 32,
        "hic_resolution": 10000,
        "dna_window": 524288,
        "reverse_complement": True,
        "max_jitter": 0,
        "genome": "hg38",
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
        "lora_scale": 0.02,
        "lora_norm": "layer",
    }
    config = BorzoiCorigamiInferencer.make_config(**config)
    inferencer = BorzoiCorigamiInferencer(config)

    # dataloader = inferencer.get_test_dataloader(batches=1, as_torch=True, return_regions=True)

    # # %%
    # for batch_id, batch in enumerate(dataloader):
    #     print(batch_id, batch)
    #     break

    # # %%
    # for key, value in batch.items():
    #     print(key, value.shape)
    # %%
    inferencer.infer_visualize()

# %%
if __name__ == "__main__":
    main()

# %%
