#%%
import torch
from bolero.tl.model.borzoi.model_lora import BorzoiLoRA
from bolero.tl.model.borzoi_human.module_hic import Corigami
from torchinfo import summary
from torch.amp import autocast

# %%
fold_id = 0
config = {
    "emb_input_features": 50,
    "base_checkpoint_path": f"/home/hanliu/data/240729-WMBRNAModel/05.borzoi/borzoi_pretrain/torch_checkpoints/borzoi.human.f{fold_id}.pt",
    "kv_bottleneck": None
}

borzoi = BorzoiLoRA.create_from_config(config).to("cuda")
borzoi.convert_to_lora()
borzoi.to('cuda')

corigami = Corigami()
corigami.to('cuda')

dna_one_hot = torch.randint(0, 2, (2, 4, 524288)).half().to("cuda")
cell_embedding = torch.randn(2, 50).float().to("cuda")

with torch.no_grad():
    atac_count, dna_embedding = borzoi.forward(x=dna_one_hot, embedding=cell_embedding, return_dna_embedding=True, crop=False)
    atac_log = torch.log1p(atac_count)
    corigami_input = torch.cat([dna_embedding, atac_log], dim=1)
    corigami(corigami_input)

# optional, freeze borzoi and only train corigami
for params in borzoi.parameters():
    params.requires_grad = False

with autocast("cuda"):
    for _ in range(100):
        atac_count, dna_embedding = borzoi.forward(x=dna_one_hot, embedding=cell_embedding, return_dna_embedding=True, crop=False)
        atac_log = torch.log1p(atac_count)
        corigami_input = torch.cat([dna_embedding, atac_log], dim=1)
        corigami(corigami_input)
