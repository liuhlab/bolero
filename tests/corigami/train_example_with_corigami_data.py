#%%
import numpy as np
from glob import glob

from bolero.tl.generic.train_helper import corigami_hg38_splits
from bolero import init
from bolero.tl.model.corigami.train import CorigamiTrainer
#%%
init(num_cpus=32, object_store_memory_gb=400, verbose=True)

#%%
indir = '/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/hic_matrix/'
cool_paths = np.sort(glob(f'{indir}*.cool'))

#%%
atac_paths = ['/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/atac.bw']
ctcf_paths = ['/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/ctcf_log2fc.bw']

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
    "bed": '/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/train.bed',
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
    "chrom_split": corigami_hg38_splits[0],
    "max_epochs": 80,
    "patience": 80,
    "train_batches": 2000,
    "val_batches": 50,
    "std": 0.1,
    "lr": 0.002,
    "use_ema": False,
    # save data
    "output_dir": "", # NEED TO CHANGE
    "wandb_project": "", # NEED TO CHANGE
    "wandb_job_type": "train",
    "wandb_group": None,
    "savename": "base",
}
config = CorigamiTrainer.make_config(**config)

#%%
trainer = CorigamiTrainer(config)
#%%
trainer.train()
