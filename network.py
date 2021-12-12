import torch.nn as nn

# network architecture
# fully connected network
class EITNet(nn.Module): 
    def __init__(self, input_width, layer_width):
        super(EITNet, self).__init__()
        self.linear_in = nn.Linear(input_width, layer_width)
        self.linear1 = nn.Linear(layer_width, layer_width)
        self.linear2 = nn.Linear(layer_width, layer_width)
        self.linear3 = nn.Linear(layer_width, layer_width)
        self.linear4 = nn.Linear(layer_width, layer_width)
        self.linear5 = nn.Linear(layer_width, layer_width)
        self.linear6 = nn.Linear(layer_width, layer_width)
        self.linear_out = nn.Linear(layer_width, 1)

        self.acti = nn.SiLU()

    def forward(self, x):
        y = self.linear_in(x) 
        y = y + self.acti(self.linear2(self.acti(self.linear1(y)))) 
        y = y + self.acti(self.linear4(self.acti(self.linear3(y)))) 
        y = y + self.acti(self.linear6(self.acti(self.linear5(y)))) 
        output = self.linear_out(y) 
        return output