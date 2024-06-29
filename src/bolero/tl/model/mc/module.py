from torch import nn

from bolero.tl.model.generic.module import Conv1dWrapper


class OutputHead(nn.Module):
    def __init__(self, n_filters=1024, kernel_size=1, out_channels=1, bias=True):
        """
        Make the final prediction of the model.

        Parameters
        ----------
        n_filters: int
            number of input channels
        kernel_size: int
            kernel size
        out_channels: int
            number of output channels
        bias: bool
            whether to use bias
        """
        super().__init__()
        self.conv_layer = Conv1dWrapper(
            in_channels=n_filters,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

    def forward(self, X, *args, output_len=None, modes=None, **kwargs):
        """Forward pass of the model."""
        X_score = self.conv_layer(X, *args, modes=modes, **kwargs)
        if output_len is None:
            trim = 0
        else:
            output_len_needed_in_X = int(output_len)
            trim = (X_score.shape[-1] - output_len_needed_in_X) // 2
        if trim > 0:
            X_score = X_score[..., trim:-trim]
        return X_score
