import torch.nn as nn

from bolero.tl.model.scprinter.module import Conv1dWrapper


class scFootprintBPNetOrigin(nn.Module):
    """
    This defines the bindingnet model
    """

    def __init__(
        self,
        dna_cnn_model=None,
        hidden_layer_model=None,
        profile_cnn_model=None,
        dna_len=2114,
        output_len=1000,
        a_embedding_dim=None,
        b_embedding_dim=None,
        example_a_embedding=None,
        rank=8,
        hidden_dim=None,
        lora_dna_cnn=False,
        lora_dilated_cnn=False,
        lora_pff_cnn=False,
        lora_output_cnn=False,
        lora_count_cnn=False,
        n_lora_layers=0,
    ):
        super().__init__()
        self.dna_cnn_model = dna_cnn_model
        self.hidden_layer_model = hidden_layer_model
        self.profile_cnn_model = profile_cnn_model
        self.dna_len = dna_len
        self.output_len = output_len

        from bolero.tl.model.scprinter.module import Conv1dLoRA

        if lora_dna_cnn:
            self.dna_cnn_model.conv = Conv1dLoRA(
                self.dna_cnn_model.conv,
                A_embedding_dim=a_embedding_dim,
                B_embedding_dim=b_embedding_dim,
                example_a_embedding=example_a_embedding,
                r=rank,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )

        hidden_layers = self.hidden_layer_model.layers
        for i in range(len(hidden_layers)):
            if lora_dilated_cnn:
                hidden_layers[i].module.conv1 = Conv1dLoRA(
                    hidden_layers[i].module.conv1,
                    A_embedding_dim=a_embedding_dim,
                    B_embedding_dim=b_embedding_dim,
                    example_a_embedding=example_a_embedding,
                    r=rank,
                    hidden_dim=hidden_dim,
                    n_layers=n_lora_layers,
                )
            if lora_pff_cnn:
                hidden_layers[i].module.conv2 = Conv1dLoRA(
                    hidden_layers[i].module.conv2,
                    A_embedding_dim=a_embedding_dim,
                    B_embedding_dim=b_embedding_dim,
                    example_a_embedding=example_a_embedding,
                    r=rank,
                    hidden_dim=hidden_dim,
                    n_layers=n_lora_layers,
                )

        if lora_output_cnn:
            self.profile_cnn_model.conv_layer = Conv1dLoRA(
                self.profile_cnn_model.conv_layer,
                A_embedding_dim=a_embedding_dim,
                B_embedding_dim=b_embedding_dim,
                example_a_embedding=example_a_embedding,
                r=rank,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )
        if isinstance(self.profile_cnn_model.linear, nn.Linear):
            print("translating linear into conv1d")
            weight = self.profile_cnn_model.linear.weight.data
            print(weight.shape)
            bias = self.profile_cnn_model.linear.bias.data
            self.profile_cnn_model.linear = Conv1dWrapper(
                weight.shape[1], weight.shape[0], 1
            )
            print(self.profile_cnn_model.linear.conv.weight.shape)
            self.profile_cnn_model.linear.conv.weight.data = weight.unsqueeze(-1)
            self.profile_cnn_model.linear.conv.bias.data = bias

        if lora_count_cnn:
            self.profile_cnn_model.linear = Conv1dLoRA(
                self.profile_cnn_model.linear,
                A_embedding_dim=a_embedding_dim,
                B_embedding_dim=b_embedding_dim,
                example_a_embedding=example_a_embedding,
                r=1,
                hidden_dim=hidden_dim,
                n_layers=n_lora_layers,
            )

    def forward(
        self, X, cell_embedding=None, region_embedding=None, output_len=None, modes=None
    ):
        """Forward pass of the model"""
        if output_len is None:
            output_len = self.output_len
        # get the motifs!
        X = self.dna_cnn_model(
            X, a_embedding=cell_embedding, b_embedding=cell_embedding
        )
        # get the hidden layer
        X = self.hidden_layer_model(
            X, a_embedding=cell_embedding, b_embedding=cell_embedding
        )
        # get the profile
        score = self.profile_cnn_model(
            X,
            a_embedding=cell_embedding,
            b_embedding=cell_embedding,
            output_len=output_len,
            modes=modes,
        )
        return score
