# %%
import numpy as np
from glob import glob
from bolero import hg38_splits
import torch
import torch.nn.functional as F
import pathlib
from skimage.transform import resize

from bolero.tl.model.corigami.train import CorigamiTrainer
from bolero.tl.model.hic.dataset import reverse_comp_hic_data_batch

# %%
# Set up
from bolero import init
init(num_cpus=48, object_store_memory_gb=320, verbose=True)

# %%
# If only predicting one cell, just need to read one cool file
indir = '/large_experiments/zhoulab/project/seqmodel/data/HBA_3C_majortype_hg38/cool_e/'
cool_paths = np.sort(glob(f'{indir}*.E.cool'))[:3]

# %%
leg = [xx.split('::')[0].split('/')[-1].split('.')[0] for xx in cool_paths]
leg

# %%
bw_paths = [f'/large_experiments/zhoulab/hanliu/wmb/Li2023Science/old_annot/bigwig/{ct}.bw' for ct in leg]
assert all([pathlib.Path(p).exists() for p in bw_paths])

# %%
config = {
    # dataset
    "cool_paths": cool_paths[:1].tolist(), # for test, only use the first cool file
    "bigwig_paths": bw_paths[:1],
    "resolution": 10000,
    "balance": False,
    "genome": 'hg38',
    "dna": False,
    "batch_size": 8,
    # trainng
    "mode": "base",
    "chrom_split": hg38_splits[0],
    "max_epochs": 80,
    "patience": 20,
    "train_batches": None,
    "val_batches": None,
    # save data
    "output_dir": "corigami_result",
    "wandb_project": "corigami_example",
    "wandb_job_type": "train",
    "wandb_group": None,
    "savename": "base",
}

config = CorigamiTrainer.make_config(**config)
trainer = CorigamiTrainer(config)

# %%
# Read data
dataset = trainer.dataset
dataset.train()
dataloader = dataset.get_dataloader(chroms=hg38_splits[0]['train'])

# %%
for batch_id, batch in enumerate(dataloader):
    batch['bw_values'] = batch['bw_values'][:, 0, :]
    genome = dataset.genome
    # Note: for corigami encoder, it has function to move feature dimesnion forward, so here we keep the feature dimension as the last dimension
    dna_one_hot = genome.get_regions_one_hot(batch['region'])
    curr_dna_length = dna_one_hot.shape[1]
    expected_dna_length = 500*(2**13)
    if expected_dna_length > curr_dna_length:
        raise ValueError(f"Expected DNA length {expected_dna_length} is longer than current DNA length {curr_dna_length}.")
    else:
        radius = (curr_dna_length - expected_dna_length) // 2
        dna_one_hot = dna_one_hot[:, radius:-radius, :].astype(np.float32)
        batch['bw_values'] = batch['bw_values'][:, radius:-radius].astype(np.float32)

    batch["values"] = batch["values"][:, 0, :, :]
    batch["values"] = resize(
        batch["values"],
        (batch["values"].shape[0], 400, 400),
        anti_aliasing=True,
    )
    # batch['values'] = np.log(batch["values"] + 1)
    batch['dna_one_hot'] = dna_one_hot
    print(batch_id, batch)
    break

# %%
for k, v in batch.items():
    print(k, v.dtype, v.shape)

# %%
batch["dna_one_hot"] = trainer.gaussian_noise(batch['dna_one_hot'], trainer.std)
batch['bw_values'] = trainer.gaussian_noise(batch['bw_values'], trainer.std)

# %%
batch = reverse_comp_hic_data_batch(batch)

# %%
dna_one_hot = torch.from_numpy(batch['dna_one_hot'].copy())
dna_one_hot.shape

# %%
bw_values = torch.from_numpy(batch['bw_values'].copy())
bw_values.shape

# %%
X = torch.cat([dna_one_hot, bw_values.unsqueeze(2)], dim=2).to(trainer.device)

# %%
y = torch.from_numpy(batch['values'].copy()).float().to(trainer.device)
y.shape

# %%
# Training
trainer._setup_model()

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
x = trainer.model.move_feature_forward(X).float()
x.shape

# %%
pred_y = trainer.model.encoder(x)
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
checkpoint = torch.load('corigami_07_26_new/base.base.best_checkpoint.pt', map_location=torch.device('cuda'))
trainer._setup_model()
model = trainer.model
optimizer = trainer._get_optimizer()
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
loss = checkpoint['best_val_loss']
print(f"Loaded model with best validation loss: {loss}")
model.eval()
# %%
batch['dna_one_hot'].shape

# %%
pred_y = model(batch['dna_one_hot'].to(trainer.device))
pred_y.shape

# %%
from skimage.transform import resize
y = batch['values'][:, 0, :, :]
y = resize(y, (8, 400, 400), anti_aliasing=True)
y.shape
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('default')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'Helvetica'

# %%
plt.imshow(y[-1], cmap='bwr', vmin=-2, vmax=2)

# %%
plt.imshow(pred_y[-1].cpu().detach().numpy(), cmap='bwr', vmin=-2, vmax=2)

# %%
np.amin(pred_y[-1].cpu().detach().numpy())
# %%
np.amax(pred_y[-1].cpu().detach().numpy())
# %%
