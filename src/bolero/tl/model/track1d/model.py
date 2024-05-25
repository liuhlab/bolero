from torch import nn
from bolero.tl.model.generic.module import DNA_CNN, DilatedCNN
from bolero.tl.model.track1d.module import OutputHead


class DialatedCNNTrack1DModel(nn.Module):
    """Predicting 1 to N 1-D genome tracks."""

    default_config = {
        "n_filters": 1024,
        "bottleneck_size": 1024,
        "dna_kernel_size": 21,
        "hidden_kernel_size": 3,
        "output_kernel_size": 1,
        "input_channels": 4,
        "output_channels": 1,
        "dna_len": 'auto',
        "output_len": 1000,
        "activation": "gelu",
        "conv_groups": 8,
        "hidden_conv_blocks": 8,
        "dilation_func": None,
        "batch_norm": True,
        "batch_norm_momentum": 0.1,
        # whether to residual connect hidden layers' input with hidden layers' output
        "bipass_connect": False,
    }

    @classmethod
    def get_default_config(cls):
        return cls.default_config

    @classmethod
    def create_from_config(cls, config: dict):
        activation = config["activation"]
        if activation.lower() == "gelu":
            activation = nn.GELU()
        else:
            activation = nn.ReLU()

        dna_cnn_model = DNA_CNN(
            n_filters=config["n_filters"],
            kernel_size=config["dna_kernel_size"],
            activation=activation,
            in_channels=config["input_channels"],
        )

        dilation_func = config['dilation_func']
        if dilation_func is None:
            dilation_func = lambda x: 2 ** (x + 1)
        hidden_layer_model = DilatedCNN(
            n_filters=config["n_filters"],
            bottleneck=config["bottleneck_size"],
            kernel_size=config["hidden_kernel_size"],
            n_blocks=config["hidden_conv_blocks"],
            groups=config["conv_groups"],
            activation=activation,
            batch_norm=config["batch_norm"],
            batch_norm_momentum=config["batch_norm_momentum"],
            dilation_func=dilation_func,
            bipass_connect=config["bipass_connect"],
        )

        output_model = OutputHead(
            n_filters=config["n_filters"],
            kernel_size=config["output_kernel_size"],
            out_channels=config["output_channels"],
            bias=True,
        )

        dna_len = config["dna_len"]
        output_len = config["output_len"]
        hidden_kernel_size = config["hidden_kernel_size"]
        hidden_conv_blocks = config["hidden_conv_blocks"]
        if dna_len == "auto":
            # calculate the dna_len to prevent the padding issue
            dna_len = output_len + dna_cnn_model.conv.weight.shape[2] - 1
            for i in range(hidden_conv_blocks):
                dna_len = dna_len + 2 * (hidden_kernel_size // 2) * dilation_func(i)
        return cls(
            dna_cnn_model=dna_cnn_model,
            hidden_layer_model=hidden_layer_model,
            output_model=output_model,
            dna_len=dna_len,
            output_len=output_len,
        )

    def __init__(
        self,
        dna_cnn_model: DNA_CNN = None,
        hidden_layer_model: DilatedCNN = None,
        output_model: OutputHead = None,
        dna_len: int = 1840,
        output_len: int = 1000,
    ):
        # ===============
        # Initialize the model
        # ===============
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.output_model = output_model
        self.dna_len = dna_len
        self.output_len = output_len

    def forward(self, X, output_len=None, modes=None, **kwargs):
        """
        Forward pass of the model.

        Parameters
        ----------
            X: The input tensor.
            output_len: The length of the output.
            modes: The modes tensor.
            kwargs: placeholder for additional keyword arguments to allow for compatibility with other models.

        Returns
        -------
            torch.Tensor: The output tensor.
        """
        if output_len is None:
            output_len = self.output_len

        # get the motifs
        X = self.dna_cnn_model(X)

        # get the hidden layer
        X = self.hidden_layer_model(X)

        # get the profile
        score = self.output_model(X, output_len=output_len, modes=modes)
        return score
