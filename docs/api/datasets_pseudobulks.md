# Datasets & Pseudobulks

How single cells become conditioned training/inference units: SEACells metacells, multi-level
categorical grouping, paired pseudobulking for delta/velocity training, the parquet build, and
the DuckDB-backed random-access store.

## Metacells & pseudobulks

::: bolero.tl.pseudobulk.seacell.run_meta_cells

::: bolero.tl.pseudobulk.tree_group.prepare_multi_level_categorical_groups

::: bolero.tl.pseudobulk.paired_pseudobulk.EnsemblePairedPseudobulker

## Data build & storage

::: bolero.pp.snap_adata.CSRRowMerge

::: bolero.pp.snap_adata.AdataPseudobulkMerger

::: bolero.pp.ray_chunk_dataset.GenomeChunkDatasetGenerator

::: bolero.tl.dataset.parquet_db.GenomeParquetDB

::: bolero.tl.dataset.sc_transforms.compressed_bytes_to_array
