# bolero

**Predicting cell-state-specific gene regulation from DNA sequence.**

Bolero is a cell-state-conditioned sequence-to-function model. It takes a 524,288 bp
one-hot DNA sequence **plus** an atlas-scale cell-state embedding (and optional conditioning
such as tissue, developmental age, or a TF-activity score) and predicts cell-state-specific
chromatin accessibility and transcript abundance at 32 bp resolution (16,384 output bins).

Under the hood it is a **frozen Borzoi/Flashzoi backbone with per-layer conditional LoRA
adapters** whose low-rank weights are generated on the fly from the cell-state embedding, so
every cell state gets its own effective network. For a chosen cell state those adapters can
be "collapsed" into a plain DNA→track model for fast inference and base-level attribution.
Bolero is trained on **Bolero-10M**: 10.8M cells across 36 datasets and 6 mammals.

## Getting started

New here? Start with [Installation](installation.md) — one `pixi install` brings up the full
GPU stack. Then work through the tutorials below in order; each is a runnable notebook with
committed outputs.

## Tutorials

A guided series that reproduces the paper's workflow end to end, from raw single cells to a
trained model and its predictions.

**Embedding & metacells** — build the cell-state representation that conditions the model.

- [01. Cell embedding](tutorials/embedding_and_meta_cell/01_cell_embedding.ipynb)
- [02. Meta cells](tutorials/embedding_and_meta_cell/02_meta_cell.ipynb)

**Meta cell AnnData & Parquet** — turn fragments into the coverage database and pseudobulks.

- [03. Meta cell AnnData](tutorials/meta_cell_adata_and_parquet/03_meta_cell_adata.ipynb)
- [04. Parquet dataset](tutorials/meta_cell_adata_and_parquet/04_parquet_dataset.ipynb)
- [05. Pseudobulks & reference signal](tutorials/meta_cell_adata_and_parquet/05_pseudobulk_and_reference.ipynb)

**Model training** — fit Bolero on one dataset or the full multi-dataset atlas.

- [06. Single dataset](tutorials/model_training/06_train_borzoi_lora.ipynb)
- [07. Multi-dataset (ATAC)](tutorials/model_training/07_train_multi_dataset_atac.ipynb)
- [08. Multi-dataset (+ gene)](tutorials/model_training/08_train_multi_dataset_gene.ipynb)

**Prediction & variant effect** — accessibility, caQTL/eQTL scoring, and DNA attribution.

- [09. Predicting accessibility](tutorials/prediction/09_prediction_task.ipynb)
- [10. Variant effect (caQTL & eQTL)](tutorials/prediction/10_qtl_task.ipynb)
- [11. DNA attribution](tutorials/prediction/11_attribution_task.ipynb)

**Score model (Bolero-Score)** — condition on a chromVAR TF-activity score (e.g. AP-1).

- [12. chromVAR TF activity](tutorials/score_model/12_chromvar_score.ipynb)
- [13. Train the score model](tutorials/score_model/13_train_score_model.ipynb)
- [14. Predict with the score model](tutorials/score_model/14_predict_score_model.ipynb)

**Cross-species** — run the trained atlas on any species' genome, DNA-only, no retraining.

- [15. Cross-species prediction](tutorials/cross_species/15_cross_species_prediction.ipynb)

## Beyond the model

`bolero` pairs with [`bolerodata`](https://github.com/liuhlab/bolerodata), a lightweight
registry that maps short keys to the datasets, trained model zoo, and QTL collections behind
the paper. It is installed automatically alongside `bolero`.

## Citation

If you use Bolero, please cite *"Bolero: predicting cell-state-specific gene regulation from
DNA sequence"* (Hanqing Liu et al.). Full citation details will be added on publication.
