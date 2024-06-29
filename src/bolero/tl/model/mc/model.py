from bolero.tl.model.track1d.model import DialatedCNNTrack1DModel


class MultiTrackmCModel(DialatedCNNTrack1DModel):
    default_config = DialatedCNNTrack1DModel.default_config.copy()
    default_config.update(
        {
            "output_channels": "REQUIRED",
            "output_len": 1000,
        }
    )
