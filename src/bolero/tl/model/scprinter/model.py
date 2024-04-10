import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scprinter.seq.minimum_footprint import multiscaleFoot
from scprinter.seq.Models import scFootprintBPNet as _scFootprintBPNet
from scprinter.seq.Models import validation_step_footprint
from tqdm import trange

from bolero.tl.model.scprinter.footprint import get_dispmodel
from bolero.utils import try_gpu


class scFootprintBPNet(_scFootprintBPNet):
    """scFootprintBPNet model for training on pseudobulk ATAC data."""

    def __init__(self, config):
        super().__init__(**config)
        self.config = config

    def fit(
        self,
        training_data,
        validation_data=None,
        validation_size=None,
        max_epochs: int = 100,
        optimizer=None,
        scheduler=None,
        validation_freq: int = 100,
        early_stopping: int = 5,
        early_stopping_tol: float = 0.001,
        return_best: bool = True,
        savename: str = "model",
        modes: np.ndarray = np.arange(2, 101, 1),
        downsample=None,
        ema=None,
        use_amp: bool = True,
        accumulate_grad: int = 1,
        batch_size=None,
        **kwargs,
    ):
        """
        This is the fit function for BPNet

        Parameters
        ----------
        training_data : TrainingData
            The training data.
        validation_data : ValidationData, optional
            The validation data, by default None.
        validation_size : int, optional
            The size of the validation data, by default None.
        max_epochs : int, optional
            The maximum number of epochs to train, by default 100.
        optimizer : Optimizer, optional
            The optimizer to use for training, by default None.
        scheduler : Scheduler, optional
            The scheduler to use for adjusting the learning rate, by default None.
        validation_freq : int, optional
            The frequency at which to perform validation, by default 100.
        early_stopping : int, optional
            The number of epochs to wait for early stopping, by default 5.
        early_stopping_tol : float, optional
            The tolerance for early stopping, by default 0.001.
        return_best : bool, optional
            Whether to return the best model, by default True.
        savename : str, optional
            The name to save the model, by default "model".
        modes : np.ndarray, optional
            The modes, by default np.arange(2, 101, 1).
        downsample : float, optional
            The downsample rate, by default None.
        ema : EMA, optional
            The Exponential Moving Average (EMA) model, by default None.
        use_amp : bool, optional
            Whether to use Automatic Mixed Precision (AMP), by default True.
        accumulate_grad : int, optional
            The number of gradient accumulation steps, by default 1.
        batch_size : int, optional
            The batch size, by default None.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        int
            The number of epochs trained.
        list
            The history of training and validation losses.
        """
        dispmodel = get_dispmodel(try_gpu())

        batch_size = training_data.batch_size if batch_size is None else batch_size
        early_stopping_counter = 0
        best_loss = np.inf
        index_all = list(np.arange(2, 101, 1))
        select_index = None
        random_modes = modes
        device = next(self.parameters()).device
        loss_history = []
        assert (
            validation_size is not None or validation_data is not None
        ), "Either validation_size or validation_data should be provided"

        if validation_data is None:
            validation_data = training_data
        if validation_size is None:
            validation_size = len(validation_data)
        if validation_freq is None:
            validation_freq = len(training_data)
        if use_amp:
            print("Using amp")

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        for epoch in range(max_epochs):
            bar = trange(
                validation_freq,
                desc=f" - (Training) {epoch + 1}",
                leave=False,
                dynamic_ncols=True,
            )
            moving_avg_loss = 0
            iteration = 0

            training_data_epoch_loader = training_data.resample()
            for data in training_data_epoch_loader:
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
                ):
                    random_modes = np.random.permutation(modes)[:30]
                    select_index = torch.as_tensor(
                        [index_all.index(mode) for mode in random_modes]
                    )
                    if len(data) == 2:
                        X, y = data
                        cell = None
                        norm_cov = None
                    else:
                        X, y, cell, peak, norm_cov = data
                        X = X[:batch_size]
                        y = y[:batch_size]
                        cell = cell[:batch_size, 0]
                        cell = cell.to(device)
                        norm_cov = None

                    X = X.to(device)
                    y = y.to(device)

                    atac = y[:, 0]
                    if downsample is not None:
                        atac = F.dropout(atac, 1 - downsample, training=self.training)
                    bias = y[:, 1]

                    footprints = multiscaleFoot(atac, bias, random_modes, dispmodel)
                    mask = ~torch.isnan(footprints)
                    if norm_cov is not None:
                        coverage = norm_cov
                    else:
                        coverage = y[:, 0].sum(dim=-1)
                        coverage = torch.log1p(coverage)

                    pred_score, pred_coverage = self.forward(
                        X, cell, modes=select_index
                    )

                    desc_str = f" - (Training) {epoch + 1}"

                    loss_footprint = F.mse_loss(pred_score[mask], footprints[mask])
                    desc_str += f" Footprint Loss: {loss_footprint.item():.2f}"

                    loss_coverage = F.mse_loss(coverage, pred_coverage)
                    desc_str += f" Coverage Loss: {loss_coverage.item():.2f}"

                    loss = (loss_footprint + loss_coverage) / accumulate_grad
                    if np.isnan(loss.item()):
                        ema, optimizer, scaler = self.load_train_state_dict(
                            ema, optimizer, scaler, savename
                        )
                        continue

                scaler.scale(loss).backward()
                moving_avg_loss += loss_footprint.item()
                if (iteration + 1) % accumulate_grad == 0:
                    scaler.unscale_(
                        optimizer
                    )  # Unscale gradients for clipping without inf/nan gradients affecting the model

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    if ema:
                        ema.update()
                    if scheduler is not None:
                        scheduler.step()
                bar.set_description(desc_str)
                bar.update(1)
                iteration += 1
                if iteration >= validation_freq:
                    break

            print(f" - (Training) {epoch + 1} Loss: {moving_avg_loss / iteration:.2f}")
            print("Learning rate", optimizer.param_groups[0]["lr"])

            bar.close()
            self.eval()

            val_loss, profile_pearson, across_pearson = validation_step_footprint(
                self, validation_data, validation_size, dispmodel, modes
            )

            val_loss_all = np.sum(val_loss)
            if np.isnan(val_loss_all):
                print("Nan loss, load last OK-ish checkpoint")
                ema, optimizer, scaler = self.load_train_state_dict(
                    ema, optimizer, scaler, savename
                )
            print(
                f" - (Validation) {epoch + 1} \
                        Loss: {val_loss_all:.5f}"
            )
            print("Profile pearson", profile_pearson)
            print("Across peak pearson", across_pearson)

            if ema:
                ema.eval()
                ema.ema_model.eval()
                val_loss, profile_pearson, across_pearson = validation_step_footprint(
                    ema.ema_model, validation_data, validation_size, dispmodel, modes
                )
                if np.sum(val_loss) > val_loss_all:
                    # ema not converged yet:
                    early_stopping_counter = 0

                val_loss_all = np.sum(val_loss)

                if np.isnan(val_loss_all):
                    print("Nan loss, load last OK-ish checkpoint")
                    ema, optimizer, scaler = self.load_train_state_dict(
                        ema, optimizer, scaler, savename
                    )
                print(
                    f" - (Validation) {epoch + 1} \
                Loss: {val_loss_all:.5f}"
                )
                print("EMA Profile pearson", profile_pearson)
                print("EMA Across peak pearson", across_pearson)
                ema.train()

            self.train()

            loss_history.append([moving_avg_loss / iteration, val_loss_all])

            if val_loss_all < (best_loss - early_stopping_tol):
                print("best loss", val_loss_all)
                best_loss = val_loss_all
                early_stopping_counter = 0
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": self.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema else None,
                }
                torch.save(checkpoint, savename)
                torch.save(self, savename + ".model.pt")
                if ema:
                    torch.save(ema, savename + ".ema.pt")
                    torch.save(ema.ema_model, savename + ".ema_model.pt")
            else:
                early_stopping_counter += 1
            if early_stopping:
                if early_stopping_counter >= early_stopping:
                    print("Early stopping")
                    break

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/train_loss": moving_avg_loss / iteration,
                        "val/val_loss": val_loss_all,
                        "val/best_val_loss": best_loss,
                        "val/profile_pearson": profile_pearson,
                        "val/across_pearson_footprint": across_pearson[0],
                        "val/across_pearson_coverage": across_pearson[1],
                        "epoch": epoch,
                    }
                )

        if return_best:
            self.load_state_dict(torch.load(savename)["state_dict"])
            print("loaded best model")

        return epoch, loss_history
