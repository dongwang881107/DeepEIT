import torch

def relative_l2_error(pred, exact):
    return torch.sqrt(torch.sum((pred-exact)**2))/torch.sqrt(torch.sum(exact**2))

def l2_error(pred, exact):
    return torch.sqrt(torch.sum(abs(pred-exact)))

def l2_error(pred, exact):
    return torch.sqrt(torch.sum((pred-exact)**2))

def mean_squared_error(pred, exact):
    return torch.sqrt(torch.sum((pred-exact)**2))/pred.size()[0]

def compute_metric(pred, exact):
    return relative_l2_error(pred, exact), l2_error(pred, exact)