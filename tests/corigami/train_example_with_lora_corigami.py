#%%
from glob import glob
import numpy as np
import pandas as pd
import scanpy as sc

from bolero.tl.generic.train_helper import corigami_hg38_splits
from bolero import init
from bolero.tl.model.corigami.train import CorigamiTrainer, CorigamiLoraTrainer
#%%
init(num_cpus=32, object_store_memory_gb=500, verbose=True)

#%%
indir = '/large_storage/zhoulab/project/seqmodel/data/HBA_3C_majortype_hg38/cool_e/'
# cool_paths = [np.sort(glob(f'{indir}*.E.cool')).tolist()[5]]
cool_paths = [f'{indir}L6_IT.E.cool']

#%%
# atac_paths = ['/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/atac.bw']
# ctcf_paths = ['/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/ctcf_log2fc.bw']
leg = [xx.split('::')[0].split('/')[-1].split('.')[0] for xx in cool_paths]
atac_paths = [f'/large_storage/zhoulab/hanliu/wmb/Li2023Science/old_annot/bigwig/{ct}.bw' for ct in leg]

# # %%
# indir = '/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/hic_matrix/'
# cool_paths = np.sort(glob(f'{indir}*.cool'))

# %%
# %%
adata = sc.read_h5ad('/large_storage/zhoulab/tlgallent/data/cell_29000_rna_raw_gencode_adata_with_embeddings.h5ad')
scvi_embedding = adata.obsm['X_scVI']
# Create a DataFrame with embeddings and cell types
df = pd.DataFrame(scvi_embedding, index=adata.obs.index)
df['cell_type'] = adata.obs['MajorType']
# Group by cell type
grouped = df.groupby('cell_type').mean()
# Get embedding
recalculated_embedding = grouped.to_numpy()

# %%
leg_map = {item: index for index, item in enumerate(grouped.index.to_list())}
leg_map

# #%%
config = {
    # dataset
    "cool_paths": cool_paths,
    "atac_paths": atac_paths,
    "ctcf_paths": None,
    "resolution": 10000,
    "balance": False,
    "genome": 'hg38',
    "batch_size": 16,
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
    "max_epochs": 80,
    "patience": 80,
    "train_batches": 500,
    "val_batches": 50,
    "std": 0.1,
    "lr": 0.00001,
    "use_ema": False,
    "plot_vmin": -2,
    "plot_vmax": 2,
    "clip_grad_norm": 0.1,
    # save data
    "output_dir": "corigami_10_09_finetune_L6_IT_from_L23_IT_conditional_lora", # NEED TO CHANGE
    "wandb_project": "corigami_10_09_finetune_L6_IT_from_L23_IT_conditional_lora", # NEED TO CHANGE
    "wandb_job_type": "train",
    "wandb_name": "corigami_dna_seq_atac_brain",
    "wandb_group": None,
    "savename": "conditional_lora",
}
config = CorigamiLoraTrainer.make_config(**config)

#%%
trainer = CorigamiLoraTrainer(config)
#%%
trainer.train()
