# Training

The Bolero model and its trainers. `Borzoi` is the reimplemented Flashzoi backbone;
`BorzoiLoRA` ("Bolero") adds the conditional-LoRA adapters and output heads. The trainers drive
the wandb setup → fit → test loop for single- and multi-dataset training.

## Model

::: bolero.tl.model.borzoi.model.Borzoi

::: bolero.tl.model.borzoi.model.model_summary

::: bolero.tl.model.borzoi.model_lora.BorzoiLoRA

## Trainers

::: bolero.tl.model.borzoi.train.BorzoiLoRATrainer

::: bolero.tl.model.borzoi.train.MultiBorzoiLoRATrainer

## Cross-validation splits

`bolero.hg38_splits` and `bolero.mm10_splits` (re-exported from
`bolero.tl.generic.train_helper`) are 5-element lists giving the Borzoi fold-0…4 chromosome
splits. Each element is a dict with `"train"`, `"valid"`, and `"test"` chromosome lists:

```python
import bolero

fold0 = bolero.hg38_splits[0]
fold0["test"]   # test chromosomes for fold 0
fold0["valid"]  # validation chromosomes for fold 0
```
