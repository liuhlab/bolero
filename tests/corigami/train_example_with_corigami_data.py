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
init(num_cpus=32, object_store_memory_gb=320, verbose=True)

#%%
indir = '/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/hic_matrix/'
cool_paths = np.sort(glob(f'{indir}*.cool'))

#%%
bw_paths = ['/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/atac.bw']

#%%
config = {
    # dataset
    "cool_paths": cool_paths[:1].tolist(),
    "bigwig_paths": bw_paths[:1],
    "resolution": 10000,
    "balance": False,
    "genome": 'hg38',
    "batch_size": 8,
    "window_size": 2097152,
    "step": 40000,
    "bed": '/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/train.bed',
    "standard_length": 2097152,
    "dna_fifth_channel": True,
    # model
    "image_scale": 256,
    # trainng
    "mode": "base",
    "chrom_split": hg38_splits[0],
    "max_epochs": 40,
    "patience": 30,
    "train_batches": 600,
    "val_batches": 50,
    "std": 0.1,
    "lr": 0.002,
    # save data
    "output_dir": "corigami_08_11",
    "wandb_project": "corigami_08_11",
    "wandb_job_type": "train",
    "wandb_group": None,
    "savename": "base",
}
config = CorigamiTrainer.make_config(**config)

#%%
trainer = CorigamiTrainer(config)
#%%
trainer.train()
