import torch
from deepeit.gradients import *

# class to generate points
class Data(object):
    def __init__(self, num_points, xmin, xmax, dim=2):
        super().__init__()
        self.num_points = num_points
        self.xmin = xmin
        self.xmax = xmax
        self.dim = dim

    def random_interior_points(self):
        # hypercubic domain
        points = self.xmin+(self.xmax-self.xmin)*torch.rand(self.num_points, self.dim)
        points.requires_grad = True
        return points

    def uniform_interior_points(self):
        # hypercubic domain
        x = torch.linspace(self.xmin, self.xmax, self.num_points)
        meshes = torch.meshgrid([x]*self.dim)
        points = torch.stack(meshes, dim=self.dim).reshape(self.num_points**self.dim, self.dim)
        points.requires_grad = True
        return points

    def random_boundary_points(self):
        # hypercubic boundary
        # hypercubic of n-1 dimension in n-dimensional space has 2n sides
        points = torch.rand(self.num_points*self.dim*2, self.dim)
        for i in range(self.dim):
            points[(2*i)*self.num_points:(2*i+1)*self.num_points,i] = torch.ones(self.num_points)*self.xmin
            points[(2*i+1)*self.num_points:(2*i+2)*self.num_points,i] = torch.ones(self.num_points)*self.xmax
        points.requires_grad = True
        return points

