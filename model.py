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
    def __init__(self, num_channels, acti):
        super().__init__()
        self.num_channels = num_channels
        self.acti = get_acti(acti)
        self.blocks = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            get_acti(acti),
            nn.Linear(num_channels, num_channels),
        )

    def forward(self, x):
        residual = x 
        x = self.blocks(x)
        x += residual
        x = self.acti(x)
        return x

# network architecture
# stack residual blocks one after another
class ResNet(nn.Module):
    def __init__(self, num_channels, num_blocks, acti, dim=2, block=ResBlock):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(dim, num_channels),
            *[block(num_channels, acti) for _ in range(num_blocks)],
            nn.Linear(num_channels, 1)
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
