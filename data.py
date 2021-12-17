import torch
import numpy as np
import sys
from torch.utils.data import Dataset

# class supervised points
class SupervisedPoints(Dataset):
    def __init__(self, num_points, lower, upper, mode='train', dim=2):
        super().__init__()
        self.num_points = num_points
        self.lower = lower
        self.upper = upper
        self.mode = mode
        self.dim = dim
        self.points = self.generate_supervised_points()
        self.solutions = u(self.points)

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return self.points[idx], self.solutions[idx]

    def generate_supervised_points(self):
        if self.mode =='train':   
            return self.lower+(self.upper-self.lower)*torch.rand(self.num_points, self.dim)
        elif self.mode == 'test':
            x1 = torch.linspace(self.lower, self.upper, self.num_points)
            x2 = torch.linspace(self.lower, self.upper, self.num_points)
            [X1,X2] = torch.meshgrid(x1, x2)
            x1 = X1.reshape(self.num_points*self.num_points, 1)
            x2 = X2.reshape(self.num_points*self.num_points, 1)
            return torch.cat((x1,x2),1)
        else:
            print('train | test')
            sys.exit(0)

# generate random training points inside the domain Omega
# square domain [0,1]x[0,1]
def generate_interior_points(num_points, dim=2):
    return torch.rand(num_points, dim)

# generate random training points on the boundary partial Omega
# square boundary x1=0, x1=1, x2=0 and x2=1
def generate_boundary_points(num_points, dim=2):
    b_left = torch.rand(num_points,dim)
    b_left[:,0] = torch.zeros(num_points)
    b_right = torch.rand(num_points,dim)
    b_right[:,0] = torch.ones(num_points)
    b_bottom = torch.rand(num_points,dim)
    b_bottom[:,1] = torch.zeros(num_points)
    b_top = torch.rand(num_points,dim)
    b_top[:,1] = torch.ones(num_points)
    return b_left,b_right,b_bottom,b_top

# generate supervised points inside the domain Omega_0
# square domain [lower,upper]x[lower,upper]
def generate_supervised_points(num_points, lower, upper, dim=2):   
    return lower+(upper-lower)*torch.rand(num_points, dim)

# ground truth solution of u(x)
# u(x) = sin(pi*x1)*sin(pi*x2)
def u(x):   
    return torch.sin(np.pi*x[:,0]).reshape(-1,1) * torch.sin(np.pi*x[:,1]).reshape(-1,1)

# ground truth solution of f(x)
# computed according to u(x)
# f(x) = (2*pi*pi+1)*sin(pi*x1)*sin(pi*x2)
def f(b):             
    return (2*np.pi**2+1)*torch.sin(np.pi*b[:,0]).reshape(-1,1) * torch.sin(np.pi*b[:,1]).reshape(-1,1)

# ground truth Neumann boundary condition g(x)
# computed according to u(x) and boundary 
# when x is on the left   boundary x1=0, the outer normal vector is (-1,0), thus g(x) = -pi*cos(pi*x1)*sin(pi*x2)
# when x is on the right  boundary x1=1, the outer normal vector is (1, 0), thus g(x) =  pi*cos(pi*x1)*sin(pi*x2)
# when x is on the bottom boundary x2=0, the outer normal vector is (0,-1), thus g(x) = -pi*sin(pi*x1)*cos(pi*x2)
# when x is on the top    boundary x2=1, the outer normal vector is (0, 1), thus g(x) =  pi*sin(pi*x1)*cos(pi*x2)
def g_neumann(b_left, b_right, b_bottom, b_top):
    g_left   = -np.pi*torch.cos(np.pi*b_left[:,0]).reshape(-1,1) * torch.sin(np.pi*b_left[:,1]).reshape(-1,1)
    g_right  =  np.pi*torch.cos(np.pi*b_right[:,0]).reshape(-1,1) * torch.sin(np.pi*b_right[:,1]).reshape(-1,1)
    g_bottom = -np.pi*torch.sin(np.pi*b_bottom[:,0]).reshape(-1,1) * torch.cos(np.pi*b_bottom[:,1]).reshape(-1,1)
    g_top    =  np.pi*torch.sin(np.pi*b_top[:,0]).reshape(-1,1) * torch.cos(np.pi*b_top[:,1]).reshape(-1,1)
    return torch.cat((g_left,g_right,g_bottom,g_top),1)

# gradient of ground truth Neumann boundary condition g(x)
# computed according to g(x)
def g_neumann_grad(b_left, b_right, b_bottom, b_top):
    g_left_grad_x1 =  np.pi**2*torch.sin(np.pi*b_left[:,0]).reshape(-1,1,1) * torch.sin(np.pi*b_left[:,1]).reshape(-1,1,1)
    g_left_grad_x2 = -np.pi**2*torch.cos(np.pi*b_left[:,0]).reshape(-1,1,1) * torch.cos(np.pi*b_left[:,1]).reshape(-1,1,1)
    g_left_grad = torch.cat((g_left_grad_x1,g_left_grad_x2),1)
   
    g_right_grad_x1 = -np.pi**2*torch.sin(np.pi*b_right[:,0]).reshape(-1,1,1) * torch.sin(np.pi*b_right[:,1]).reshape(-1,1,1)
    g_right_grad_x2 =  np.pi**2*torch.cos(np.pi*b_right[:,0]).reshape(-1,1,1) * torch.cos(np.pi*b_right[:,1]).reshape(-1,1,1)
    g_right_grad = torch.cat((g_right_grad_x1,g_right_grad_x2),1)

    g_bottom_grad_x1 = -np.pi**2*torch.cos(np.pi*b_bottom[:,0]).reshape(-1,1,1) * torch.cos(np.pi*b_bottom[:,1]).reshape(-1,1,1)
    g_bottom_grad_x2 =  np.pi**2*torch.sin(np.pi*b_bottom[:,0]).reshape(-1,1,1) * torch.sin(np.pi*b_bottom[:,1]).reshape(-1,1,1)
    g_bottom_grad = torch.cat((g_bottom_grad_x1,g_bottom_grad_x2),1)

    g_top_grad_x1 =  np.pi**2*torch.cos(np.pi*b_top[:,0]).reshape(-1,1,1) * torch.cos(np.pi*b_top[:,1]).reshape(-1,1,1)
    g_top_grad_x2 = -np.pi**2*torch.sin(np.pi*b_top[:,0]).reshape(-1,1,1) * torch.sin(np.pi*b_top[:,1]).reshape(-1,1,1)
    g_top_grad = torch.cat((g_top_grad_x1,g_top_grad_x2),1)

    return torch.cat((g_left_grad,g_right_grad,g_bottom_grad,g_top_grad),2)