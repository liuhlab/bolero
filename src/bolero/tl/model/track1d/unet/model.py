from torch import nn
from .module import Conv1DBlock, Down, Up, TransformerStack


class ATACUNet(nn.Module):
    def __init__(self):
        super(ATACUNet, self).__init__()
        self.in_channels = 4
        self.n_classes = 1
        self.base_filters = 64

        self.inc = Conv1DBlock(self.in_channels, self.base_filters)
        self.down1 = Down(self.base_filters, self.base_filters * 2)
        self.down2 = Down(self.base_filters * 2, self.base_filters * 4)
        self.up1 = Up(self.base_filters * 4, self.base_filters * 2)
        self.up2 = Up(self.base_filters * 2, self.base_filters)
        self.conv = nn.Conv1d(self.base_filters, self.n_classes, kernel_size=1)
        # may adjust the kernal size.
        self.out_conv = nn.Conv1d(
            self.n_classes, self.n_classes, kernel_size=101, padding=50
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.conv(x)
        x = self.out_conv(x)
        x = x.squeeze(1)
        return x

class UNetTrans(nn.Module):
    def __init__(self):
        super(UNetTrans, self).__init__()
        self.in_channels = 4
        self.n_classes = 1
        self.base_filters = 64

        self.inc = Conv1DBlock(self.in_channels, self.base_filters)
        self.down1 = Down(self.base_filters, self.base_filters * 2)
        self.down2 = Down(self.base_filters * 2, self.base_filters * 4)

        self.transformer = TransformerStack(
            embed_size=self.base_filters * 4,
            num_heads=4,
            ff_hidden_dim=256,
            dropout=0.1,
            num_layers=4,
        )

        self.up1 = Up(self.base_filters * 4, self.base_filters * 2)
        self.up2 = Up(self.base_filters * 2, self.base_filters)
        self.conv = nn.Conv1d(self.base_filters, self.n_classes, kernel_size=1)
        # may adjust the kernal size.
        self.out_conv = nn.Conv1d(
            self.n_classes, self.n_classes, kernel_size=101, padding=50
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x3 = x3.permute(2, 0, 1)
        x3 = self.transformer(x3)
        x3 = x3.permute(1, 2, 0)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.conv(x)
        x = self.out_conv(x)
        x = x.squeeze(1)
        return x
