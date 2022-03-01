import torch
from deepeit.gradients import *

# class to determin PDE system and boundary conditions
class PDE(object):
    def __init__(self, equation, solution, inverse, xmin, xmax):
        super().__init__()
        self.equation = equation
        self.solution = solution
        self.inverse = inverse
        self.xmin = xmin
        self.xmax = xmax

    def dirichlet_bc(self, x, mode='s'):
        return self.inverse(x) if mode=='s' else self.solution(x)

    def neumann_bc(self, x, u, s):
        bn = boundary_normal(x, self.xmin, self.xmax)
        return s*(torch.sum(gradient(u,x)*bn,1).reshape(-1,1))

    def robin_bc(self, x, u, s):
        return self.dirichlet_bc_u(x) + self.neumann_bc(x, u, s)