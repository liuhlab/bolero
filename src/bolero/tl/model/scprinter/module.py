from torch import nn

from bolero.tl.generic.module import Conv1dWrapper, GenericModule


class FootprintsHead(GenericModule):
    """
    This is the output head of the footprints model, predict the multi-scale footprints and region total coverage.
    """

    default_config = {
        "n_filters": 1024,
        "output_kernel_size": 1,
        "output_scales": 99,
    }

    def __init__(
        self,
        n_filters=1024,
        output_kernel_size=1,
        output_scales=99,
    ):
        """
        Initialize the FootprintsHead module.

        Parameters
        ----------
        n_filters: int
            number of filters
        kernel_size: int
            kernel size
        output_scales: int
            number of footprints scales
        """
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = output_kernel_size
        self.n_scales = output_scales

        self.conv_layer = Conv1dWrapper(
            in_channels=self.n_filters,
            out_channels=self.n_scales,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

    def forward(self, X, *args, output_len=None, **kwargs):
        """Forward pass of the FootprintsHead module."""
        X_bindingscore = self.conv_layer(X, *args, **kwargs)

        if output_len is None:
            trim = 0
        else:
            output_len_needed_in_X = int(output_len)
            trim = (X_bindingscore.shape[-1] - output_len_needed_in_X) // 2

        if trim > 0:
            X_bindingscore = X_bindingscore[..., trim:-trim]

        return X_bindingscore

    def reset_parameters(self):
        """Reset the parameters of the module."""
        self.conv_layer.reset_parameters()


class CoverageHead(GenericModule):
    def __init__(self, n_filters=1024):
        super().__init__()
        self.n_filters = n_filters

        self.to_out = Conv1dWrapper(
            in_channels=self.n_filters,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.softplus = nn.Softplus()

    def forward(self, X, *args, **kwargs):
        """
        Forward pass of the CoverageHead module.

        X is detached and meaned over the last dimension.
        During training, this head is learnt separately from the rest of the model.
        """
        if self.training:
            X = X.detach()

        X_out = self.to_out(X, *args, **kwargs)

        cov = self.softplus(X_out).sum(dim=-1)
        return cov[..., 0]  # shape: (batch_size,)

    def reset_parameters(self):
        """Reset the parameters of the module."""
        self.to_out.reset_parameters()
