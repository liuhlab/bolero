#%%
import numpy as np
from glob import glob
from bolero import hg38_splits
from bolero.tl.model.hic.dataset import HiCTrackDataset
from bolero import init
import pathlib

from bolero.tl.model.corigami.train import CorigamiTrainer
#%%
init(num_cpus=48, object_store_memory_gb=320, verbose=True)

#%%
indir = '/large_experiments/zhoulab/project/seqmodel/data/HBA_3C_majortype_hg38/cool_e/'
cool_paths = np.sort(glob(f'{indir}*.E.cool'))[:3]

leg = [xx.split('::')[0].split('/')[-1].split('.')[0] for xx in cool_paths]
leg

bw_paths = [f'/large_experiments/zhoulab/hanliu/wmb/Li2023Science/old_annot/bigwig/{ct}.bw' for ct in leg]
assert all([pathlib.Path(p).exists() for p in bw_paths])
#%%
config = {
    # dataset
    "cool_paths": cool_paths[:1].tolist(),
    "bigwig_paths": bw_paths[:1],
    "resolution": 10000,
    "balance": False,
    "genome": 'hg38',
    "dna": False,
    "batch_size": 8,
    # trainng
    "mode": "base",
    "chrom_split": hg38_splits[0],
    "max_epochs": 1,
    "patience": 1,
    "train_batches": 10,
    "val_batches": 5,
    # save data
    "output_dir": "corigami_07_28",
    "wandb_project": "corigami_result",
    "wandb_job_type": "train",
    "wandb_group": None,
    "savename": "base",
}
config = CorigamiTrainer.make_config(**config)

#%%
trainer = CorigamiTrainer(config)
trainer.train()
