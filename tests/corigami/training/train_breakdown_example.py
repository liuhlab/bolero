# %%
import pandas as pd
import numpy as np
from glob import glob
import pathlib
import scanpy as sc
from skimage.transform import resize
import torch
import torch.nn.functional as F

from bolero.tl.generic.train_helper import corigami_hg38_splits
from bolero.tl.model.corigami.train import CorigamiTrainer, CorigamiLoraTrainer
# %%
# Set up
from bolero import init
init(num_cpus=16, object_store_memory_gb=200, verbose=True)

# %%
# If only predicting one cell, just need to read one cool file
indir = '/large_storage/zhoulab/project/seqmodel/data/HBA_3C_majortype_hg38/cool_e/'
cool_paths = np.sort(glob(f'{indir}*.E.cool')).tolist()
# cool_paths = [path for path in cool_paths if 'group' not in path]
# indir = '/large_experiments/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/hic_matrix/'
# cool_paths = np.sort(glob(f'{indir}*.cool'))

# %%
leg = [xx.split('::')[0].split('/')[-1].split('.')[0] for xx in cool_paths]
leg = [cell_type for cell_type in leg if 'group' not in cell_type]

cool_paths = [f'{indir}{cell_type}.E.cool' for cell_type in leg]


# %%
atac_paths = [f'/large_storage/zhoulab/hanliu/wmb/Li2023Science/old_annot/bigwig/{ct}.bw' for ct in leg]
# assert all([pathlib.Path(p).exists() for p in bw_paths])
# atac_paths = ['/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/atac.bw']
# ctcf_paths = ['/large_storage/zhoulab/project/seqmodel/data/corigami/corigami_data/data/hg38/imr90/genomic_features/ctcf_log2fc.bw']

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

leg_map = {item: index for index, item in enumerate(grouped.index.to_list())}
leg_map

# %%
config = {
    # dataset
    "cool_paths": cool_paths, # for test, only use the first cool file
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
    "data_1d_keys": ("atac", ),
    "smooth_moving_average": False,
    "kernel_size": None,
    "cool_data_norm_mode": None,
    "dim_shift": False,
    "lora": False,
    # "leg_map": leg_map,
    # model
    "image_scale": 64,
    "encoder_in_channel": 5,
    "encoder_num_epi": 1,
    # "recalculated_embedding": recalculated_embedding.tolist(),
    # "preset": "classic",
    # training
    "mode": "base",
    "chrom_split": corigami_hg38_splits[0],
    "max_epochs": 80,
    "patience": 10,
    "train_batches": None,
    "val_batches": None,
    "std": 0.1,
    "lr": 0.0002,
    "use_ema": False,
    # save data
    "output_dir": "/large_storage/zhoulab/yishuang/project/bolero/tests/corigami/training/2024-11-24",
    "wandb_project": "2024_11_24_corigami",
    "wandb_job_type": "train",
    "wandb_group": None,
    "wandb_name": "corigami",
    "savename": "base",
}
# config = CorigamiLoraTrainer.make_config(**config)
# # %%
# trainer = CorigamiLoraTrainer(config)
config = CorigamiTrainer.make_config(**config)
trainer = CorigamiTrainer(config)

# %%
# Read data
dataset = trainer.dataset
dataset.train()
dataloader = dataset.get_dataloader(chroms=corigami_hg38_splits[0]['train'], as_torch=True)

# %%
for batch_id, batch in enumerate(dataloader):
    print(batch_id, batch)
    break

# %%
for k, v in batch.items():
    print(k, v.dtype, v.shape)

# %%
dna_one_hot = batch["dna_one_hot"]
dna_one_hot=dna_one_hot.half()
dna_one_hot.shape

# %%
feature_list = [batch[feat] for feat in config['data_1d_keys']]
features = torch.cat([feature.unsqueeze(1) for feature in feature_list], dim=1)
features.shape

# %%
X = torch.cat([dna_one_hot, features], dim=1)
X.shape

# %%
X += torch.randn_like(X) * trainer.std
X.shape
# %%
y = batch['values']
y.shape

# %%
embedding = batch.get('embedding', None)
embedding.shape
# %%
# Training
trainer._setup_model()
model = trainer.model
model.to(trainer.device)
# %%
from torchinfo import summary
summary(trainer.model, input_data=None, depth=6)
# %%
print(model._model_summary())
# %%
trainer.device

# %%
for name, params in model.named_parameters():
	print(name, params.shape, params.requires_grad)

# %%
emb_example = torch.randint(size=(1, 1), low=0, high=trainer.model.recalculated_embedding.shape[0])
input_data = {
                "x": torch.ones(1, 6, 2097152).to(trainer.device),
                "embedding": emb_example.to(trainer.device),
            }
model.to(trainer.device)
summary(model, input_size=None, input_data=input_data, depth=5)

# %%
summary(model.encoder, input_data=None, depth=3)

# %%
summary(model.attn, input_size=[8, 256, 256], depth=2)

# %%
summary(model.decoder, input_data=None,  depth=2)

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
model.to(trainer.device)
embedding = model.cell_type_embedding(embedding).view(embedding.shape[0], trainer.model.recalculated_embedding.shape[1])
embedding.shape
# %%
# embedding = None
# pred_y = model.encoder(X, embedding=embedding)
pred_y = model.encoder(X)
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
if pred_y.shape[1] > model.image_scale:
    pred_y = model.trim_encoder_output(pred_y)
pred_y.shape

# %%
pred_y = model.move_feature_forward(pred_y)
pred_y.shape

# %%
# pred_y = model.attn(pred_y, embedding=embedding)
pred_y = model.attn(pred_y)
pred_y = model.move_feature_forward(pred_y)
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
pred_y = model.diagonalize(pred_y)

# %%
# pred_y = model.decoder(pred_y, embedding=embedding).squeeze(1)
pred_y = model.decoder(pred_y).squeeze(1)
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
checkpoint = torch.load("./corigami_09_15_atac_brain_learning_rate_batch_16_no_smooth/base.base.best_checkpoint.pt", map_location=torch.device('cuda'))
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
dataloader = dataset.get_dataloader(chroms=corigami_hg38_splits[0]['train'])

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

# %%
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
plt.imshow(y[0], cmap='coolwarm', vmin=0, vmax=0.01)

# %%
plt.imshow(pred_y[0], cmap='coolwarm', vmin=-2, vmax=2)

# %%
np.amin(pred_y[-1])
# %%
np.amax(pred_y[-1])
# %%
from bolero.pl.hic import HicExamplePlotter
plotter = HicExamplePlotter("values", "pred_")
batch["pred_"] = pred_y.detach()
fig, _ = plotter.plot(batch, figsize=(20, 20), dpi=100, top_example=2, bottom_example=2, plot_channel=0)
# %%
