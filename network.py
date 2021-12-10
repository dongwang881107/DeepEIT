import torch

# actication function
def acti(x):
    return x*torch.sigmoid(x)  

# network architecture
class DeepNet(torch.nn.Module): 
    def __init__(self, input_width, layer_width):
        super(DeepNet, self).__init__()
        self.linear_in = torch.nn.Linear(input_width, layer_width)
        self.linear1 = torch.nn.Linear(layer_width, layer_width)
        self.linear2 = torch.nn.Linear(layer_width, layer_width)
        self.linear3 = torch.nn.Linear(layer_width, layer_width)
        self.linear4 = torch.nn.Linear(layer_width, layer_width)
        self.linear5 = torch.nn.Linear(layer_width, layer_width)
        self.linear6 = torch.nn.Linear(layer_width, layer_width)
        self.linear_out = torch.nn.Linear(layer_width, 1)

    def forward(self, x):
        y = self.linear_in(x) 
        y = y + acti(self.linear2(acti(self.linear1(y)))) 
        y = y + acti(self.linear4(acti(self.linear3(y)))) 
        y = y + acti(self.linear6(acti(self.linear5(y)))) 
        output = self.linear_out(y) 
        return output