from numpy import negative
import torch.nn as nn

# actication function
def get_acti(acti):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
        ['sigmoid', nn.Sigmoid()],
        ['tanh', nn.Tanh()],
        ['selu', nn.SELU()],
        ['swish', nn.SiLU()],
        ['none', nn.Identity()] 
    ])[acti]

# residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, acti='swish'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.acti = get_acti(acti)
        self.blocks = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            get_acti(acti),
            nn.Linear(out_channels, in_channels),
        )

    def forward(self, x):
        residual = x 
        x = self.blocks(x)
        x += residual
        x = self.acti(x)
        return x

# network architecture
# stack residual blocks one after another
class EITNet(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResBlock, block_num=1):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            *[block(out_channels, out_channels) for _ in range(block_num)],
            nn.Linar(out_channels, 1)
        )

    def forward(self, x):
        x = self.blocks(x)
        return x