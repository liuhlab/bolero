from typing import Any, Dict, Tuple

import torch
from scprinter.seq.Models import scFootprintBPNet
from scprinter.seq.Modules import DNA_CNN, DilatedCNN, Footprints_head


def construct_model_from_config(config: Dict[str, Any]) -> Tuple[Any, int, int]:
    """
    Construct a model from the given configuration.

    Parameters
    ----------
        config (Dict[str, Any]): The configuration dictionary containing the model parameters.

    Returns
    -------
        Tuple[Any, int, int]: A tuple containing the constructed model, the length of the DNA sequence, and the output length.

    Raises
    ------
        None

    """
    # CNN Layer parameters
    n_filters = config["n_filters"]  # 768
    head_kernel_size = config["head_kernel_size"]  # 1
    kernel_size = config["kernel_size"]  # 3
    dilation_base = config["dilation_base"]  # 1  # Dialated CNN
    dilation_func = lambda x: 2 ** (x + dilation_base)

    groups = config["groups"]  # 8  # grouped CNNs

    # CNN Block parameters
    activation = config["activation"]  # "gelu"
    if activation == "relu":
        activation = torch.nn.ReLU()
    elif activation == "gelu":
        activation = torch.nn.GELU()
    batch_norm = config["batch_norm"]  # True
    batch_norm_momentum = config["batch_norm_momentum"]  # 0.1

    bottleneck_factor = config["bottleneck_factor"]  # 0.5 or 1, 1 means no bottleneck
    bottleneck = int(
        n_filters * bottleneck_factor
    )  # turn factor into actual number of filters in the bottleneck layer
    n_layers = config["n_layers"]  # 8, actually means blocks of layers

    rezero = config["rezero"]  # False, means use residual connection instead of rezero

    # General parameters,
    # no_inception means the model is similar to the original BPNet
    # inception_vertion 2 means using a CNN
    # Ruochi's default is always False, meaning there will be inception layers
    no_inception = config["no_inception"]
    if no_inception:
        n_inception_layers = 0
    inception_version = config["inception_version"]  # 2
    n_inception_layers = config[
        "n_inception_layers"
    ]  # 8, meaning all layer blocks are inception layers
    inception_layers_after = config["inception_layers_after"]
    # True, meaning use inception layers after the normal layers, this actually has no effect since all layers are inception layers
    if inception_layers_after:
        inception_bool = [False] * (n_layers - n_inception_layers) + [True] * (n_inception_layers)
    else:
        inception_bool = [True] * n_inception_layers + [False] * (n_layers - n_inception_layers)

    acc_dna_cnn = DNA_CNN(
        n_filters=n_filters,
    )

    acc_hidden = DilatedCNN(
        n_filters=n_filters,
        bottleneck=bottleneck,
        n_layers=n_layers,
        kernel_size=kernel_size,
        groups=groups,
        activation=activation,
        batch_norm=batch_norm,
        residual=True,
        rezero=rezero,
        dilation_func=dilation_func,
        batch_norm_momentum=batch_norm_momentum,
        inception=inception_bool,
        inception_version=inception_version,
    )

    acc_head = Footprints_head(
        n_filters, kernel_size=head_kernel_size, n_scales=99, per_peak_feats=1
    )

    output_len = 800
    dna_len = output_len + acc_dna_cnn.conv.weight.shape[2] - 1
    for i in range(n_layers):
        dna_len = dna_len + 2 * (kernel_size // 2) * dilation_func(i)
    print("dna_len", dna_len)
    print("output_len", output_len)

    acc_model = scFootprintBPNet(
        dna_cnn_model=acc_dna_cnn,
        hidden_layer_model=acc_hidden,
        profile_cnn_model=acc_head,
        dna_len=dna_len,
        output_len=output_len,
    )
    return acc_model, dna_len, output_len
