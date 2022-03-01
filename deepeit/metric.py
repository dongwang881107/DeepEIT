import torch

class Metric(object):
    def __init__(self, pred, exact):
        super().__init__()
        self.pred = pred
        self.exact = exact

    def relative_l2_error(self):
        return torch.sqrt(torch.sum((self.pred-self.exact)**2))/torch.sqrt(torch.sum(self.exact**2))

    def l1_error(self):
        return torch.sqrt(torch.sum(abs(self.pred-self.exact)))

    def l2_error(self):
        return torch.sqrt(torch.sum((self.pred-self.exact)**2))

    def mean_squared_error(self):
        return torch.sqrt(torch.sum((self.pred-self.exact)**2))/self.pred.size()[0]
