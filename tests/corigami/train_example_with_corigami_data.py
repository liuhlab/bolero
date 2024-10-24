#%%
import numpy as np
from glob import glob

from bolero.tl.generic.train_helper import corigami_hg38_splits
from bolero import init
from bolero.tl.model.corigami.train import CorigamiTrainer
#%%
init(num_cpus=32, object_store_memory_gb=400, verbose=True)

#%%
indir = '/large_storage/zhoulab/project/seqmodel/data/HBA_3C_majortype_hg38/cool_q/'
cool_paths = np.sort(glob(f'{indir}*.Q.cool'))[:1].tolist()
# cool_paths = [f'{indir}L6_IT.E.cool']

#%%
# atac_paths = ['/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/atac.bw']
# ctcf_paths = ['/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/ctcf_log2fc.bw']
leg = [xx.split('::')[0].split('/')[-1].split('.')[0] for xx in cool_paths]
atac_paths = [f'/large_storage/zhoulab/hanliu/wmb/Li2023Science/old_annot/bigwig/{ct}.bw' for ct in leg]

# # %%
# indir = '/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/hic_matrix/'
# cool_paths = np.sort(glob(f'{indir}*.cool'))
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
    "window_size": 2097152,
    "step": 40000,
    "bed": '/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/train.bed',
    "standard_length": 2097152,
    "dna_fifth_channel": True,
    "data_1d_keys": ("atac",),
    "smooth_moving_average": False,
    "kernel_size": None,
    "cool_data_norm_mode": None,
    "dim_shift": True,
    "lora": False,
    "cap_value": 0.02,
    # model
    "image_scale": 256,
    "encoder_in_channel": 5,
    "encoder_num_epi": 1,
    # trainng
    "mode": "base",
    "pretrained_model": None, # "/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/model_weights/corigami_base.ckpt",
    "chrom_split": corigami_hg38_splits[0],
    "max_epochs": 80,
    "patience": 80,
    "train_batches": 1000,
    "val_batches": 50,
    "std": 0.1,
    "lr": 0.0002,
    "use_ema": False,
    "plot_vmin": 0,
    "plot_vmax": 0.01,
    "clip_grad_norm": 1,
    # save data
    "output_dir": "corigami_10_09_ASC_cool_q_cap", # NEED TO CHANGE
    "wandb_project": "corigami_10_09_ASC_cool_q_cap", # NEED TO CHANGE
    "wandb_job_type": "train",
    "wandb_name": "corigami_dna_seq_atac_brain",
    "wandb_group": None,
    "savename": "base",
}
config = CorigamiTrainer.make_config(**config)

#%%
trainer = CorigamiTrainer(config)
#%%
trainer.train()
