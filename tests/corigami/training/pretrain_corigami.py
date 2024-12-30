#%%
import argparse
from glob import glob
import pandas as pd
import scanpy as sc

import numpy as np

from bolero.tl.generic.train_helper import corigami_hg38_splits
from bolero import init
from bolero.tl.model.corigami.train import CorigamiTrainer

init(num_cpus=32, object_store_memory_gb=360, verbose=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="corigami pretrain")
    parser.add_argument('--cell_types', nargs='+', type=str, help="The cell type to train")
    parser.add_argument('--output_dir', type=str, help="The output directory")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    cell_types = args.cell_types

    indir = '/large_storage/zhoulab/project/seqmodel/data/HBA_3C_majortype_hg38/cool_e/'

    # cool_paths = np.sort(glob(f'{indir}*.E.cool')).tolist()
    # cell_types = [xx.split('::')[0].split('/')[-1].split('.')[0] for xx in cool_paths]
    # cell_types = [cell_type for cell_type in cell_types if 'group' not in cell_type]
    cool_paths = [f'{indir}{cell_type}.E.cool' for cell_type in cell_types]

    atac_paths = [f'/large_storage/zhoulab/hanliu/wmb/Li2023Science/old_annot/bigwig/{ct}.bw' for ct in cell_types]

    #%%
    config = {
        # dataset
        "cool_paths": cool_paths,
        "atac_paths": atac_paths,
        "ctcf_paths": None,
        "resolution": 10000,
        "balance": False,
        "genome": 'hg38',
        "batch_size": 8,
        "window_size": 524288,
        "step": 40000,
        "bed": '/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/train.bed',
        "standard_length": 524288,
        "dna_fifth_channel": True,
        "data_1d_keys": ("atac",),
        "smooth_moving_average": False,
        "kernel_size": None,
        "cool_data_norm_mode": None,
        "dim_shift": False,
        "lora": False,
        # model
        "image_scale": 64,
        "encoder_in_channel": 5,
        "encoder_num_epi": 1,
        # trainng
        "mode": "base",
        "chrom_split": corigami_hg38_splits[0],
        "max_epochs": 100,
        "patience": 10,
        "train_batches":5000,
        "val_batches": 1000,
        "std": 0.1,
        "lr": 0.0002,
        "use_ema": False,
        "plot_vmin": -2,
        "plot_vmax": 2,
        "clip_grad_norm": 1,
        # save data
        "output_dir": args.output_dir, # NEED TO CHANGE
        "wandb_project": args.output_dir, # NEED TO CHANGE
        "wandb_job_type": "train",
        "wandb_name": f"pretrain_corigami_500k_{args.cell_types[0]}", # NEED TO CHANGE
        "wandb_group": None,
        "savename": "base",
    }
    config = CorigamiTrainer.make_config(**config)
    trainer = CorigamiTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
