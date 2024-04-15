import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scprinter.seq.minimum_footprint import multiscaleFoot
from scprinter.seq.Models import scFootprintBPNet as _scFootprintBPNet
from scprinter.seq.Models import validation_step_footprint
from tqdm import trange

from bolero.tl.footprint.footprint import get_dispmodel
from bolero.utils import try_gpu


class scFootprintTrainer(_scFootprintBPNet):
    """scFootprintBPNet model for training on pseudobulk ATAC data."""

    def __init__(self, config):
        super().__init__(**config)

        self.config = config
        self.patience = config.get("patience", 10)
        self.early_stopping_counter = 0

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.device = next(self.parameters()).device
        self.max_epochs = config.get("max_epochs", 100)

        self.validation_freq = config.get("validation_freq", 100)

        self.use_amp = self.config.get("use_amp", False)

    def get_scaler(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        return scaler

    def fit(
        self,
        training_dataset,
        validation_dataset,
        sample=None,
        region=None,
        local_shuffle_buffer_size=10000,
        randomize_block_order=False,
    ):
        for epoch in range(self.max_epochs):
            bar = trange(
                self.validation_freq,
                desc=" - (Training) {epoch}".format(epoch=epoch + 1),
                leave=False,
                dynamic_ncols=True,
            )
            moving_avg_loss = 0
            iteration = 0

            data_loader = training_dataset.get_dataloader(
                sample=None,
                region=None,
                local_shuffle_buffer_size=10000,
                randomize_block_order=False,
            )
            footprinter = training_dataset.get_footprinter()
            for batch in data_loader:
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.use_amp):
                    

        return
