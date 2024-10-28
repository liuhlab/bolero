# %%
import argparse
import pandas as pd
import scanpy as sc

from bolero.tl.generic.train_helper import corigami_hg38_splits
from bolero.tl.model.corigami.infer import CorigamiInferencer
from bolero import init

# Set up
init(num_cpus=16, object_store_memory_gb=200, verbose=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="corigami pretrain")
    parser.add_argument('--cell_type', type=str, help="The cell type to train")
    parser.add_argument('--checkpoint_folder', type=str, help="The path to the checkpoint folder")
    parser.add_argument('--mode', type=str, default='base', help="The mode to train")
    parser.add_argument('--no_training', action='store_true', help="Whether to train the model")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    cell_type = args.cell_type

    indir = '/large_storage/zhoulab/project/seqmodel/data/HBA_3C_majortype_hg38/cool_e/'
    cool_paths = [f'{indir}{cell_type}.E.cool']
    leg = [xx.split('::')[0].split('/')[-1].split('.')[0] for xx in cool_paths]

    atac_paths = [f'/large_storage/zhoulab/hanliu/wmb/Li2023Science/old_annot/bigwig/{ct}.bw' for ct in leg]

    adata = sc.read_h5ad('/large_storage/zhoulab/tlgallent/data/cell_29000_rna_raw_gencode_adata_with_embeddings.h5ad')
    scvi_embedding = adata.obsm['X_scVI']
    # Create a DataFrame with embeddings and cell types
    df = pd.DataFrame(scvi_embedding, index=adata.obs.index)
    df['cell_type'] = adata.obs['MajorType']
    # Group by cell type
    grouped = df.groupby('cell_type').mean()

    leg_map = {item: index for index, item in enumerate(grouped.index.to_list())}

    if args.mode == 'lora_finetune':
        inference_folder = 'plot_lora'
    elif args.mode == "conditional_lora_finetune":
        inference_folder = "plot_conditional_lora"
    else:
        if args.no_training:
            inference_folder = "plot_no_training"
        else:
            inference_folder = "plot_pretrained"

    config = {
        # dataset
        "cool_paths": cool_paths, # for test, only use the first cool file
        "atac_paths": atac_paths,
        "ctcf_paths": None,
        "resolution": 10000,
        "balance": False,
        "genome": 'hg38',
        "batch_size": 2,
        "window_size": 2097152,
        "step": 40000,
        "bed": '/large_storage/zhoulab/yishuang/project/bolero/tests/corigami/train.bed',
        "standard_length": 2097152,
        "dna_fifth_channel": True,
        "data_1d_keys": ("atac", ),
        "smooth_moving_average": False,
        "kernel_size": None,
        "cool_data_norm_mode": None,
        "dim_shift": True,
        "lora": False,
        "leg_map": leg_map,
        # model
        "image_scale": 256,
        "encoder_in_channel": 5,
        "encoder_num_epi": 1,
        "recalculated_embedding": None,
        "rank": 8,
        "alpha": 16,
        "preset": "classic",
        # training
        "mode": args.mode,
        "pretrained_model": f"/large_storage/zhoulab/yishuang/project/bolero/tests/corigami/{args.checkpoint_folder}/{args.mode}.{args.mode}.best_checkpoint.pt",
        "chrom_split": corigami_hg38_splits[0],
        "max_epochs": 80,
        "patience": 10,
        "train_batches": None,
        "val_batches": None,
        "std": 0.1,
        "lr": 3e-5,
        "weight_decay": 0,
        "use_ema": True,
        "plot_vmin": -2,
        "plot_vmax": 2,
        "clip_grad_norm": 0.1,
        "lora_dropout": 0.01,
        # save data
        "output_dir": f"/large_storage/zhoulab/yishuang/project/bolero/tests/corigami/{inference_folder}/{cell_type}",
        "wandb_project": "corigami_result",
        "wandb_job_type": "inference",
        "wandb_group": None,
        "wandb_name": "corigami",
        "savename": "base",
    }
    config = CorigamiInferencer.make_config(**config)
    inferencer = CorigamiInferencer(config)
    inferencer.infer_visualize()

if __name__ == "__main__":
    main()
