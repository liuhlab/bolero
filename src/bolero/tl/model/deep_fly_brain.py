import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFlyBrain(nn.Module):
    """DeepFlyBrain model."""

    def __init__(self, out_dims, seq_shape=(1001, 4), motif_file_path=None):
        super().__init__()

        # Define layers
        self.conv1d = nn.Conv1d(in_channels=seq_shape[0], out_channels=1024, kernel_size=24)
        self.maxpool = nn.MaxPool1d(kernel_size=12, stride=12)
        self.dropout1 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(1024, 128)  # Assuming the correct input size here
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.dense2 = nn.Linear(256, 256)  # Assuming the correct input size here
        self.dropout3 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(256, out_dims)  # Assuming the correct input size here

        # Custom weight initialization
        self.motif_file_path = motif_file_path
        if self.motif_file_path is not None:
            self.initialize_weights()

    def process_input(self, x):
        """Process input through the layers."""
        x = F.relu(self.conv1d(x))
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)  # Rearrange dimensions for TimeDistributed Dense
        x = F.relu(self.dense1(x))
        x = x.permute(0, 2, 1)  # Rearrange back after Dense
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.dense2(x))
        x = self.dropout3(x)
        return x

    def forward(self, x):
        """Forward pass."""
        # Forward input
        x_forward = self.process_input(x)

        # Reverse input
        x_reversed = x.flip(dims=[1]).flip(dims=[2])
        x_reversed = self.process_input(x_reversed)

        # Concatenate
        merged = torch.cat((x_forward, x_reversed), dim=1)

        # Final layers
        out = self.output_layer(merged)
        out = F.sigmoid(out)
        return out

    # def initialize_weights(self):
    #     """Initialize weights using the motif file."""
    #     with open(self.motif_file_path, "rb") as f:
    #         motif_dict = pickle.load(f)
    #     conv_weights = self.conv1d.weight.data
    #     # Assuming 'w' and other required variables are defined
    #     # Initialize weights here as done in the Keras model


# Usage
seq_shape = (1, 100, 100)  # Example shape
model = DeepFlyBrain(seq_shape, "/motif_file.pkl")

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()


# TensorFlow code from the original DeepFlyBrain model

# reverse_lambda_ax2 = Lambda(lambda x: K.reverse(x,axes=2))
# reverse_lambda_ax1 = Lambda(lambda x: K.reverse(x,axes=1))

# def get_output(input_layer, hidden_layers):
#     output = input_layer
#     for hidden_layer in hidden_layers:
#         output = hidden_layer(output)
#     return output

# def build_model():
#     forward_input = Input(shape=seq_shape)
#     reverse_input = Input(shape=seq_shape)

#     layer0 = [
#         Conv1D(1024, kernel_size=24, padding="valid", activation='relu', kernel_initializer='random_uniform'),
#         MaxPooling1D(pool_size=12, strides=12, padding='valid'),
#         Dropout(0.5),
#         TimeDistributed(Dense(128, activation='relu')),
#         Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
#         Dropout(0.5),
#         Flatten(),
#         Dense(256, activation='relu'),
#         Dropout(0.5),]
#     layer1 = [
#         Dense(81, activation='sigmoid')]
#     forward_output_f = get_output(forward_input, layer0)
#     reverse_output_r = get_output(reverse_lambda_ax2(reverse_lambda_ax1(forward_input)), layer0)
#     merged_output = concatenate([forward_output_f,reverse_output_r],axis=1)
#     output = get_output(merged_output, layer1)
#     model = Model(input=forward_input, output=output)

#     f = open("/motif_file.pkl", "rb")
#     motif_dict = pickle.load(f)
#     f.close()
#     conv_weights = model.layers[3].get_weights()
#     for i, name in enumerate(motif_dict):
#         conv_weights[0][int((w-len(motif_dict[name]))/2):int((w-len(motif_dict[name]))/2) + len(motif_dict[name]), :, i] = motif_dict[name]
#     model.layers[3].set_weights(conv_weights)

#     model.summary()
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model
