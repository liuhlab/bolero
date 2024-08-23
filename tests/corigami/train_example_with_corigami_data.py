#%%
import numpy as np
from glob import glob
import pathlib
from skimage.transform import resize
import torch
import torch.nn.functional as F

from bolero import hg38_splits, init
from bolero.tl.model.corigami.dataset import HiCTrackDataset
from bolero.tl.model.corigami.train import CorigamiTrainer
#%%
init(num_cpus=32, object_store_memory_gb=360, verbose=True)

#%%
indir = '/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/hic_matrix/'
cool_paths = np.sort(glob(f'{indir}*.cool'))

#%%
atac_paths = ['/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/atac.bw']
ctcf_paths = ['/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/ctcf_log2fc.bw']

#%%
config = {
    # dataset
    "cool_paths": cool_paths[:1].tolist(),
    "atac_paths": atac_paths,
    "ctcf_paths": ctcf_paths,
    "resolution": 10000,
    "balance": False,
    "genome": 'hg38',
    "batch_size": 8,
    "window_size": 2097152,
    "step": 40000,
    "bed": '/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/train.bed',
    "standard_length": 2097152,
    "dna_fifth_channel": True,
    "data_1d_keys": ("atac", "ctcf",),
    # model
    "image_scale": 256,
    "encoder_in_channel": 5,
    "encoder_num_epi": 2,
    # trainng
    "mode": "base",
    "pretrained_model": None, # "/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/model_weights/corigami_base.ckpt",
    "chrom_split": hg38_splits[0],
    "max_epochs": 50,
    "patience": 30,
    "train_batches": 5000,
    "val_batches": 50,
    "std": 0.1,
    "lr": 0.002,
    "use_ema": False,
    # save data
    "output_dir": "corigami_08_22_ema_disabled_amp_enabled_5_channel_ctcf_atac",
    "wandb_project": "corigami_08_22_ema_disabled_amp_enabled_5_channel_ctcf_atac",
    "wandb_job_type": "train",
    "wandb_group": None,
    "savename": "base",
}
config = CorigamiTrainer.make_config(**config)

#%%
trainer = CorigamiTrainer(config)
#%%
trainer.train()
