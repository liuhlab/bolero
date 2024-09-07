# %%
import numpy as np
from glob import glob
import pathlib
from skimage.transform import resize
import torch
import torch.nn.functional as F

from bolero.tl.generic.train_helper import hg38_splits
from bolero.tl.model.corigami.train import CorigamiTrainer
# %%
# Set up
from bolero import init
init(num_cpus=64, object_store_memory_gb=500, verbose=True)

# %%
# If only predicting one cell, just need to read one cool file
# indir = '/large_experiments/zhoulab/project/seqmodel/data/HBA_3C_majortype_hg38/cool_e/'
# cool_paths = np.sort(glob(f'{indir}*.E.cool'))[:3]
indir = '/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/hic_matrix/'
cool_paths = np.sort(glob(f'{indir}*.cool'))

# %%
leg = [xx.split('::')[0].split('/')[-1].split('.')[0] for xx in cool_paths]
leg

# %%
# bw_paths = [f'/large_experiments/zhoulab/hanliu/wmb/Li2023Science/old_annot/bigwig/{ct}.bw' for ct in leg]
# assert all([pathlib.Path(p).exists() for p in bw_paths])
atac_paths = ['/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/atac.bw']
ctcf_paths = ['/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/ctcf_log2fc.bw']

# %%
config = {
    # dataset
    "cool_paths": cool_paths[:1].tolist(), # for test, only use the first cool file
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
    # training
    "mode": "base",
    "chrom_split": hg38_splits[0],
    "max_epochs": 80,
    "patience": 10,
    "train_batches": None,
    "val_batches": None,
    "std": 0.1,
    "lr": 0.002,
    "use_ema": True,
    # save data
    "output_dir": "/large_storage/zhoulab/yishuang/project/bolero/tests/corigami/corigami_result",
    "wandb_project": "corigami_result",
    "wandb_job_type": "train",
    "wandb_group": None,
    "savename": "base",
}
config = CorigamiTrainer.make_config(**config)
# %%
trainer = CorigamiTrainer(config)

# %%
# Read data
dataset = trainer.dataset
dataset.train()
dataloader = dataset.get_dataloader(chroms=hg38_splits[0]['train'], as_torch=True)

# %%
for batch_id, batch in enumerate(dataloader):
    print(batch_id, batch)
    break

# %%
for k, v in batch.items():
    print(k, v.dtype, v.shape)

# %%
dna_one_hot = batch["dna_one_hot"]
dna_one_hot.shape

# %%
feature_list = [batch[feat] for feat in config['data_1d_keys']]
features = torch.cat([feature.unsqueeze(1) for feature in feature_list], dim=1)
features.shape

# %%
X = torch.cat([dna_one_hot, features], dim=1)
X.shape

# %%
y = batch['values']
y.shape

# %%
# Training
trainer._setup_model()

# %%
from torchinfo import summary
summary(trainer.model, input_size=X.shape, depth=2)

# %%
summary(trainer.model.encoder, input_size=X.shape, depth=2)

# %%
summary(trainer.model.attn, input_size=[8, 256, 256], depth=2)

# %%
summary(trainer.model.decoder, input_size=[8, 512, 256, 256],  depth=2)

# %%
# Get total, reserved, and allocated GPU memory
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r - a  # Free memory inside reserved
print(f"Total GPU memory: {t}")
print(f"Free GPU memory: {f}")
print(f"Used GPU memory: {a}")

# %%
pred_y = trainer.model.encoder(X)
pred_y.shape

# %%
# Get total, reserved, and allocated GPU memory
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r - a  # Free memory inside reserved
print(f"Total GPU memory: {t}")
print(f"Free GPU memory: {f}")
print(f"Used GPU memory: {a}")

# %%
if pred_y.shape[1] > trainer.model.image_scale:
    pred_y = trainer.model.trim_encoder_output(pred_y)
pred_y.shape

# %%
pred_y = trainer.model.move_feature_forward(pred_y)
pred_y.shape

# %%
pred_y = trainer.model.attn(pred_y)
pred_y = trainer.model.move_feature_forward(pred_y)
pred_y.shape

# %%
# Get total, reserved, and allocated GPU memory
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r - a  # Free memory inside reserved
print(f"Total GPU memory: {t}")
print(f"Free GPU memory: {f}")
print(f"Used GPU memory: {a}")

# %%
pred_y = trainer.model.diagonalize(pred_y)
pred_y = trainer.model.decoder(pred_y).squeeze(1)
pred_y.shape

# %%
# Get total, reserved, and allocated GPU memory
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r - a  # Free memory inside reserved
print(f"Total GPU memory: {t}")
print(f"Free GPU memory: {f}")
print(f"Used GPU memory: {a}")

# %%
loss_ = F.mse_loss(pred_y, y.to(trainer.device))
loss_

# %%
# Inference
checkpoint = torch.load("corigami_base.ckpt", map_location=torch.device('cuda'))
trainer._setup_model()
model = trainer.model
model.to(trainer.device)
# %%
model.eval()
model_weights = checkpoint['state_dict']
for key in list(model_weights):
    model_weights[key.replace('model.', '')] = model_weights.pop(key)
# %%
model.load_state_dict(model_weights)

# %%
# optimizer = trainer._get_optimizer()
# optimizer.load_state_dict(checkpoint['optimizer'])
# loss = checkpoint['best_val_loss']
# print(f"Loaded model with best validation loss: {loss}")
# epoch_info = torch.load("corigami_08_06/base.base.epoch_info.pt")
# print(epoch_info)


# %%
dataset = trainer.dataset
dataset.eval()
dataloader = dataset.get_dataloader(chroms=hg38_splits[0]['valid'])

# %%
for batch_id, batch in enumerate(dataloader):
    print(batch_id, batch)
    break

# %%
dna_one_hot = batch["dna_one_hot"]
feature_list = [batch[feat] for feat in config['data_1d_keys']]
features = torch.cat([feature.unsqueeze(1) for feature in feature_list], dim=1)
X = torch.cat([dna_one_hot, features], dim=1)
X.shape
# %%
pred_y = model(X.to(trainer.device))
pred_y.shape

# %%
y = batch['values'].cpu().detach().numpy()
pred_y = pred_y.cpu().detach().numpy()

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'Helvetica'

# %%
from matplotlib.colors import LinearSegmentedColormap
color_map = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])

# %%
plt.imshow(y[6], cmap='coolwarm', vmin=0, vmax=5)

# %%
plt.imshow(pred_y[6], cmap='coolwarm', vmin=0, vmax=5)

# %%
np.amin(pred_y[-1])
# %%
np.amax(pred_y[-1])

# %%
np.amin(normalized_pred_y[-2])
# %%
np.amax(normalized_pred_y[-2])
# %%
from bolero.pl.hic import HicExamplePlotter
plotter = HicExamplePlotter("values", "pred_")
batch["pred_"] = pred_y.detach()
fig, _ = plotter.plot(batch, figsize=(20, 20), dpi=100, top_example=2, bottom_example=2, plot_channel=0)
# %%
