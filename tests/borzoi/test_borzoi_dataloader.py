# %%
from bolero.tl.model.borzoi_human.dataset import BorzoiDatasetOnline
from bolero import init

init(num_cpus=32, object_store_memory_gb=200)

# %%
cell_types = ['ASC', 'Vip', 'L5_IT']
dataset_path = '/large_storage/zhoulab/project/seqmodel/data/bolero_data_path.json'
embeddings_path = '/large_storage/zhoulab/project/seqmodel/data/cell_type_embeddings.json'
# %%
borzoi_online_dataset = BorzoiDatasetOnline(cell_types=cell_types, dataset_path=dataset_path, embeddings_path=embeddings_path, bed=None, genome="hg38", use_borzoi_regions=True, resolution=10000)
borzoi_online_dataset.train()
# %%
(train_folds, valid_folds, test_folds, train_regions, valid_regions, test_regions) = borzoi_online_dataset.get_train_valid_test(fold=0)
loader1 = borzoi_online_dataset.get_dataloader(region_bed=train_regions)

# %%
bed = borzoi_online_dataset.bed
bed
# %%
for batch_id, batch in enumerate(loader1):
    print(batch_id, batch)
    break
# %%
for k, v in batch.items():
    print(k, v.dtype, v.shape)
# %%
