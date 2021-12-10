import torch

# generate random training points from Omega
def generate_interior_points(num_points, dim=2):
    return torch.rand(num_points, dim)

# generate random training points from partial Omega
def generate_boundary_points(num_points, dim=2):
    xb1 = torch.rand(num_points, dim)
    xb1[:, 0] = torch.zeros(num_points)
    xb2 = torch.rand(num_points, dim)
    xb2[:, 0] = torch.ones(num_points)
    xb3 = torch.rand(num_points, dim)
    xb3[:, 1] = torch.zeros(num_points)
    xb4 = torch.rand(num_points, dim)
    xb4[:, 1] = torch.ones(num_points)
    return torch.cat((xb1,xb2,xb3,xb4),1)

# generate supervised points from Omega_0
def interior_data(num_points):   
    x1 = 0.5*torch.ones(num_points,1)+0.25*torch.rand(num_points,1)
    x2 = 0.5*torch.ones(num_points,1)+0.25*torch.rand(num_points,1)
    return torch.cat((x1,x2),1)



# exact solution of f(x)
# f(x) = (2*pi*pi+1)*sin(pi*x1)*sin(pi*x2)
def f_exact(x):             
    return (2*torch.pi**2+1)*torch.sin(torch.pi*x[:,0]).reshape(-1,1)*torch.sin(torch.pi*x[:,1]).reshape(-1,1)

# exact solution of u(x)
# u(x) = sin(pi*x1)*sin(pi*x2)
def u_exact(x):   
    return torch.sin(torch.pi*x[:,0]).reshape(-1,1)*torch.sin(torch.pi*x[:,1]).reshape(-1,1)

# exact Neumann boundary g(x)
def g(xb):
    g_xb1 = -pi*torch.cos(pi*xb1[:,0]).reshape(-1,1)*torch.sin(pi*xb1[:,1]).reshape(-1,1)
    g_xb2 = pi*torch.cos(pi*xb2[:,0]).reshape(-1,1)*torch.sin(pi*xb2[:,1]).reshape(-1,1)
    g_xb3 = -pi*torch.sin(pi*xb3[:,0]).reshape(-1,1)*torch.cos(pi*xb3[:,1]).reshape(-1,1)
    g_xb4 = pi*torch.sin(pi*xb3[:,0]).reshape(-1,1)*torch.cos(pi*xb3[:,1]).reshape(-1,1)

    
    g_xb1_x1 = pi**2*torch.sin(pi*xb1[:,0]).reshape(-1,1)*torch.sin(pi*xb1[:,1]).reshape(-1,1)
    g_xb1_x2 = -pi**2*torch.cos(pi*xb1[:,0]).reshape(-1,1)*torch.cos(pi*xb1[:,1]).reshape(-1,1)
    
   
    g_xb2_x1 = -pi**2*torch.sin(pi*xb2[:,0]).reshape(-1,1)*torch.sin(pi*xb2[:,1]).reshape(-1,1)
    g_xb2_x2 = pi**2*torch.cos(pi*xb2[:,0]).reshape(-1,1)*torch.cos(pi*xb2[:,1]).reshape(-1,1)

    g_xb3_x1 = -pi**2*torch.cos(pi*xb3[:,0]).reshape(-1,1)*torch.cos(pi*xb3[:,1]).reshape(-1,1)
    g_xb3_x2 = pi**2*torch.sin(pi*xb3[:,0]).reshape(-1,1)*torch.sin(pi*xb3[:,1]).reshape(-1,1)

    g_xb4_x1 = pi**2*torch.cos(pi*xb4[:,0]).reshape(-1,1)*torch.cos(pi*xb4[:,1]).reshape(-1,1)
    g_xb4_x2 = -pi**2*torch.sin(pi*xb4[:,0]).reshape(-1,1)*torch.sin(pi*xb4[:,1]).reshape(-1,1)
