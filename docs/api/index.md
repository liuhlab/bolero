# API Reference

This reference documents the **user-facing** public API of `bolero` — the symbols the
tutorials use and the paper figures reach through the
[`bolerodata`](https://github.com/liuhlab/bolerodata) companion. Import by fully-qualified
module path (only the top-level package and `pp` curate re-exports).

The pages are grouped to mirror the tutorial workflow:

| Page | What it covers | Tutorials |
|---|---|---|
| [Genome & Sequence](genome_sequence.md) | Assembly handles, DNA one-hot, region utilities | all |
| [Datasets & Pseudobulks](datasets_pseudobulks.md) | Metacells, pseudobulking, parquet build, streaming datasets | 01–05 |
| [Training](training.md) | The Bolero model (`BorzoiLoRA`) and its trainers | 06–08, 13 |
| [Prediction & Variant Effect](prediction.md) | Inference tasks, caQTL/eQTL, directed evolution | 09–11, 14, 15 |
| [Interpretation](interpretation.md) | chromVAR, motif scanning, TF-MoDISco, finemo, TomTom | 12; Fig 5 |
| [Plotting](plotting.md) | Attribution sequence logos, IGV browser | 11; Fig 1b/2 |

!!! note
    This is a research code base that carries experimental/vestigial code; the pages here
    intentionally document only the finalized, published surface. See the tutorials for
    end-to-end, runnable examples.
