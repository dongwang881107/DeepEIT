import torch
from deepeit.data import *

# class of loss function
# needs PDE and Model
class Loss(object):
    def __init__(self, pde, model_u, model_s):
        super().__init__()
        self.pde = pde
        self.model_u = model_u
        self.model_s = model_s

    def interior_loss(self, x):
        u_pred = self.model_u(x)
        s_pred = self.model_s(x)
        pred = self.pde.equation(x, u_pred, s_pred)
        u_exact = self.pde.solution(x)
        s_exact = self.pde.inverse(x)
        exact = self.pde.equation(x, u_exact, s_exact)
        return torch.sum((pred-exact)**2)/x.size()[0]
    
    def dirichlet_boundary_loss(self, x, mode='s'):
        pred = self.model_s(x) if mode=='s' else self.model_u(x)
        exact = self.pde.dirichlet_bc(x, mode)
        return torch.sum((pred-exact)**2)/x.size()[0]
    
    def neumann_boundary_loss(self, x):
        u_pred = self.model_u(x)
        s_pred = self.model_s(x)
        pred = self.pde.neumann_bc(x, u_pred, s_pred)
        u_exact = self.pde.solution(x)
        s_exact = self.pde.inverse(x)
        exact = self.pde.neumann_bc(x, u_exact, s_exact)
        return torch.sum((pred-exact)**2)/x.size()[0]

    def robin_boundary_loss(self, x):
        return self.dirichlet_boundary_loss(x, 's') + self.neumann_boundary_loss(x)

    def observe_loss(self, x, noisy_data):
        pred = self.model_u(x)
        return torch.sum((pred-noisy_data)**2)/x.size()[0]

    def g_loss(self, x, g_func):
        u_pred = self.model_u(x)
        g_exact = g_func(x)
        loss_lap = torch.sum(laplacian(u_pred,x)**2)/x.size()[0]
        loss_grad = torch.sum(gradient(u_pred-g_exact,x)**2)/x.size()[0]
        loss_ori = torch.sum((u_pred-g_exact)**2)/x.size()[0]
        return loss_lap, loss_grad, loss_ori

    def l2_loss(self, x):
        s_pred = self.model_s(x)
        return torch.sum(s_pred**2)/x.size()[0]

    def h1_loss(self, x):
        s_pred = self.model_s(x)
        loss_grad = torch.sum(gradient(s_pred,x)**2)/x.size()[0]
        loss_ori = torch.sum(s_pred**2)/x.size()[0]
        return loss_grad, loss_ori

    def tv_loss(self, x):
        s_pred = self.model_s(x)
        grad_s = gradient(s_pred,x)
        return torch.sum(torch.abs(grad_s))/x.size()[0]