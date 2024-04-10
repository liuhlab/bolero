import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFlyBrain(nn.Module):
    """DeepFlyBrain model."""

    def __init__(self, out_dims, seq_shape=(500, 4), motif_db=None):
        super.__init__()

        # Define layers
        self.conv1d = nn.Conv1d(
            in_channels=seq_shape[1], out_channels=1024, kernel_size=24
        )
        self.maxpool = nn.MaxPool1d(kernel_size=12, stride=12)
        self.dropout1 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(1024, 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.dense2 = nn.Linear(256 * 39, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(512, out_dims)

        self.motif_db = motif_db
        if self.motif_db is not None:
            self.update_conv1d_weights()

    def process_input(self, x):
        """Process input through the layers."""
        # Rearrange dimensions to batch, bases/channels, sequence positions
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1d(x))
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.dense1(x))
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.dense2(x))
        x = self.dropout3(x)
        return x

    def update_conv1d_weights(self):
        """
        Update the weights of the conv1d layer based on the motifs in the motif database.

        Returns
        -------
            None

        Raises
        ------
            None
        """
        # Get the default initialized weights
        init_weights = self.conv1d.weight.data.clone()
        kernel_size = init_weights.shape[2]

        for out_channel_idx, motif in enumerate(self.motif_db.motifs):
            pwm = motif.pwm
            # pwm.shape == (motif_length, base)

            pwm = torch.from_numpy(pwm.T.values).to(
                dtype=self.conv1d.weight.dtype, device=self.conv1d.weight.device
            )
            pwm_size = pwm.shape[1]
            pad = int((kernel_size - pwm_size) / 2)
            init_weights[out_channel_idx, :, pad : pad + pwm_size] = pwm

        # Assign the modified weights back
        self.conv1d.weight.data = init_weights

    def forward(self, x):
        """Forward pass."""
        # Forward input
        x_forward = self.process_input(x)

        # Reverse input
        x_reversed_complemented = x.flip(dims=[1]).flip(dims=[2])
        x_reversed_complemented = self.process_input(x_reversed_complemented)

        # Concatenate
        merged = torch.cat((x_forward, x_reversed_complemented), dim=1)

        # Final layers
        out = self.output_layer(merged)
        return out
