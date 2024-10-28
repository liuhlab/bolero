import argparse
import pandas as pd
import scanpy as sc

from bolero.tl.generic.train_helper import corigami_hg38_splits
from bolero import init
from bolero.tl.model.corigami.train import CorigamiLoraTrainer
init(num_cpus=32, object_store_memory_gb=500, verbose=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="corigami lora finetune")
    parser.add_argument('--cell_types', nargs='+', type=str, help="The cell type to train")
    parser.add_argument('--output_dir', nargs='+', type=str, help="The output directory")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    cell_types = args.cell_types

    indir = '/large_storage/zhoulab/project/seqmodel/data/HBA_3C_majortype_hg38/cool_e/'
    cool_paths = [f'{indir}{cell_type}.E.cool' for cell_type in cell_types]

    leg = [xx.split('::')[0].split('/')[-1].split('.')[0] for xx in cool_paths]
    atac_paths = [f'/large_storage/zhoulab/hanliu/wmb/Li2023Science/old_annot/bigwig/{ct}.bw' for ct in leg]

    adata = sc.read_h5ad('/large_storage/zhoulab/tlgallent/data/cell_29000_rna_raw_gencode_adata_with_embeddings.h5ad')
    scvi_embedding = adata.obsm['X_scVI']
    # Create a DataFrame with embeddings and cell types
    df = pd.DataFrame(scvi_embedding, index=adata.obs.index)
    df['cell_type'] = adata.obs['MajorType']
    # Group by cell type
    grouped = df.groupby('cell_type').mean()
    # Get embedding
    recalculated_embedding = grouped.to_numpy()
    leg_map = {item: index for index, item in enumerate(grouped.index.to_list())}

    config = {
        # dataset
        "cool_paths": cool_paths,
        "atac_paths": atac_paths,
        "ctcf_paths": None,
        "resolution": 10000,
        "balance": False,
        "genome": 'hg38',
        "batch_size": 8,
        "window_size": 2097152,
        "step": 40000,
        "bed": '/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/train.bed',
        "standard_length": 2097152,
        "dna_fifth_channel": True,
        "data_1d_keys": ("atac",),
        "smooth_moving_average": False,
        "kernel_size": None,
        "cool_data_norm_mode": None,
        "dim_shift": False,
        "lora": True,
        "leg_map": leg_map,
        # model
        "image_scale": 256,
        "encoder_in_channel": 5,
        "encoder_num_epi": 1,
        "recalculated_embedding": recalculated_embedding.tolist(),
        "rank": 8,
        "alpha": 16,
        "preset": "all_conditional",
        # training
        "mode": "conditional_lora_finetune",
        "pretrained_model": "/large_storage/zhoulab/yishuang/project/bolero/tests/corigami/corigami_10_07_L23_IT_pretrain/base.base.best_checkpoint.pt",
        "chrom_split": corigami_hg38_splits[0],
        "max_epochs": 40,
        "patience": 20,
        "train_batches": 1000,
        "val_batches": 50,
        "std": 0.1,
        "lr": 3e-5,
        "weight_decay": 0,
        "use_ema": False,
        "plot_vmin": -2,
        "plot_vmax": 2,
        "clip_grad_norm": 0.1,
        "lora_dropout": 0.1,
        # save data
        "output_dir": "corigami_10_21_finetune_all_from_L23_IT_conditional_lora", # NEED TO CHANGE
        "wandb_project": "corigami_10_21_finetune_all_from_L23_IT_conditional_lora", # NEED TO CHANGE
        "wandb_job_type": "train",
        "wandb_name": "corigami_dna_seq_atac_brain",
        "wandb_group": None,
        "savename": "conditional_lora",
    }
    config = CorigamiLoraTrainer.make_config(**config)
    trainer = CorigamiLoraTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
