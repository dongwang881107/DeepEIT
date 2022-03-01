import sys
import torch.nn as nn
import torch.optim as optim

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

def set_optim(model_u, model_s, args):
    optimizer = optim.Adam([{'params':model_u.parameters(),'lr':args.lr_u},\
         {'params':model_s.parameters(),'lr':args.lr_s}])
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_iters, gamma=args.gamma)
    elif args.scheduler == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=args.gamma, total_iters = args.decay_iters)
    elif args.scheduler == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.decay_iters, eta_min=1e-5)
    elif args.scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1000)
    else:
        print('step | linear | exp | cos | reduce')
        sys.exit(0)
    return optimizer, scheduler

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

# FCN block
class FCNBlock(nn.Module):
    def __init__(self, num_channels, acti):
        super().__init__()
        self.num_channels = num_channels
        self.acti = get_acti(acti)
        self.blocks = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            get_acti(acti),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

# EITNet
# stack blocks together
class EITNet(nn.Module):
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
