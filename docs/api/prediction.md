# Prediction & Variant Effect

The inference engine and its high-level **task** methods, the variant-effect managers, and the
directed-evolution / enhancer-design engine.

`BorzoiPredictor` owns a trained model plus a data manager and exposes the task methods that map
to the paper (`prediction_task`, `caqtl_task`, `eqtl_task`, `peak_task`, `attribution_task`, …).
`BorzoiSignalPredictor` is the signal/flow variant used across the prediction tutorials.

## Predictors

::: bolero.tl.predict.predictor_borzoi.BorzoiPredictor

::: bolero.tl.predict.predictor_borzoi.BorzoiSignalPredictor

::: bolero.tl.predict.predictor_borzoi.BorzoiInputXGradient

## Variant-effect managers

::: bolero.tl.predict.task_manager.caQTLManager

::: bolero.tl.predict.task_manager.eQTLManager

::: bolero.tl.predict.task_manager.PeakManager

## Directed evolution & enhancer design

::: bolero.tl.predict.dna_gen.DNASynthesisFactory

::: bolero.tl.predict.dna_gen.DNAEvolutionFactory

## Evaluation utilities

::: bolero.tl.predict.utils.multi_level_peak_stats
