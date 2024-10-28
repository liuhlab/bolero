## Corigami
Here we adapt the [corigami model](https://github.com/tanjimin/C.Origami/blob/main/src/corigami/model/corigami_models.py) to predict Hi-C using DNA sequence and ATAC data. To predict the Hi-C on a specific cell type, we tried four different modes,
- Training from scratch on the target cell type
- Inference on the target cell type by updating the ATAC data using a model pretrained on a different cell type
- Lora finetune on the target cell type given a model pretrained on a different cell type
- Conditional Lora finetune on the target cell type given a model pretrained on a different cell type

## Running the code
### Training
1. Install the packages needed
```
git clone https://github.com/lhqing/bolero.git
cd bolero
conda env create -f environment.yaml
pip install -e ".[dev,test]"
```

2. Train the model from scratch
Take L6_IT for example
```
cd tests/corigami/training
python pretrain_corigami.py --cell_type L6_IT --output_dir L6_IT_pretrain
```

3. Lora finetune the model
Take L6_IT for example
```
cd tests/corigami/training
python lora_finetune_corigami.py --cell_type L6_IT --output_dir L6_IT_lora_finetune
```

4. Conditional Lora Finetune the model
```
cd tests/corigami/training
python lora_finetune_corigami.py --cell_types L4_IT L6_IT Amy ASC L6_CT L56_NP MGC MSN --output_dir conditional_lora_fientune
```

### Inference
1. To get the predicted hic value, there are two modes: base or lora_finetune

Train from scratch:
```
cd tests/corigami/inference
python inference_corigami.py --cell_type L6_IT --mode base --checkpoint_folder L6_IT_pretrain
```

Inference on a pretrained model
```
cd tests/corigami/inference
python inference_corigami.py --cell_type L6_IT --mode base --checkpoint_folder L23_IT_pretrain --no_training
```

Lora finetune on a pretrained model
```
cd tests/corigami/inference
python inference_corigami.py --cell_type L6_IT --mode lora_finetune --checkpoint_folder L6_IT_lora_finetune
```

Conditional Lora Finetune on a pretrained model
```
cd tests/corigami/inference
python inference_corigami.py --cell_type L6_IT --mode conditional_lora_finetune --checkpoint_folder conditional_lora_finetune
```

2. Visualiza the differential loop heatmap
Run `diff_loop_heatmap.ipynb`

3. Visualize the hic map for a specific region

a. provide a temporary region under `tests/corigami/inference/temp.bed`
b. run visualization code

Visualize on the result trained from scratch
`python inference_hic_plot.py --cell_type L6_IT --checkpoint_folder L6_IT_pretrain --mode base`

Visualize on the result inferencing on a pretrained model
`python inference_hic_plot.py --cell_type L6_IT --checkpoint_folder L23_IT_pretrain --mode base --no_training`

Visualize on the result fine-tuning on a pretrained model
`python inference_hic_plot.py --cell_type L6_IT --mode lora_finetune --checkpoint_folder L6_IT_lora_finetune`

Visualize on the result conditional fine-tuning on a pretrained model
`python inference_hic_plot.py --cell_type L6_IT --mode conditional_lora_finetune --checkpoint_folder conditional_lora_finetune`
