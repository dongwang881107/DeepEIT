import torch

# generate random training points inside the domain Omega
# square domain [0,1]x[0,1]
def generate_interior_points(num_points, dim=2):
    return torch.rand(num_points, dim)

# generate random training points on the boundary partial Omega
# square boundary x1=0, x1=1, x2=0 and x2=1
def generate_boundary_points(num_points, dim=2):
    b_left = torch.rand(num_points,dim,1)
    b_left[:,0,0] = torch.zeros(num_points)
    b_right = torch.rand(num_points,dim,1)
    b_right[:,0,0] = torch.ones(num_points)
    b_bottom = torch.rand(num_points,dim,1)
    b_bottom[:,1,0] = torch.zeros(num_points)
    b_top = torch.rand(num_points,dim,1)
    b_top[:,1,0] = torch.ones(num_points)
    return torch.cat((b_left,b_right,b_bottom,b_top),2)

# generate supervised points inside Omega_0
# square domain [0.5,0.75]x[0.5,0.75]
def interior_data(num_points):   
    x1 = 0.5*torch.ones(num_points,1)+0.25*torch.rand(num_points,1)
    x2 = 0.5*torch.ones(num_points,1)+0.25*torch.rand(num_points,1)
    return torch.cat((x1,x2),1)

# ground truth solution of u(x)
# u(x) = sin(pi*x1)*sin(pi*x2)
def u(x):   
    return torch.sin(torch.pi*x[:,0]).reshape(-1,1) * torch.sin(torch.pi*x[:,1]).reshape(-1,1)

# ground truth solution of f(x)
# computed according to u(x)
# f(x) = (2*pi*pi+1)*sin(pi*x1)*sin(pi*x2)
def f(x):             
    return (2*torch.pi**2+1)*torch.sin(torch.pi*x[:,0]).reshape(-1,1) * torch.sin(torch.pi*x[:,1]).reshape(-1,1)

# ground truth Neumann boundary condition g(x)
# computed according to u(x) and boundary 
# when x is on the left   boundary x1=0, the outer normal vector is (-1,0), thus g(x) = -pi*cos(pi*x1)*sin(pi*x2)
# when x is on the right  boundary x1=1, the outer normal vector is (1, 0), thus g(x) =  pi*cos(pi*x1)*sin(pi*x2)
# when x is on the bottom boundary x2=0, the outer normal vector is (0,-1), thus g(x) = -pi*sin(pi*x1)*cos(pi*x2)
# when x is on the top    boundary x2=1, the outer normal vector is (0, 1), thus g(x) =  pi*sin(pi*x1)*cos(pi*x2)
def g(b):
    g_left   = -torch.pi*torch.cos(torch.pi*b[:,0,0]).reshape(-1,1) * torch.sin(torch.pi*b[:,1,0]).reshape(-1,1)
    g_right  =  torch.pi*torch.cos(torch.pi*b[:,0,1]).reshape(-1,1) * torch.sin(torch.pi*b[:,1,1]).reshape(-1,1)
    g_bottom = -torch.pi*torch.sin(torch.pi*b[:,0,2]).reshape(-1,1) * torch.cos(torch.pi*b[:,1,2]).reshape(-1,1)
    g_top    =  torch.pi*torch.sin(torch.pi*b[:,0,3]).reshape(-1,1) * torch.cos(torch.pi*b[:,1,3]).reshape(-1,1)
    return torch.cat((g_left,g_right,g_bottom,g_top),1)

# gradient of ground truth Neumann boundary condition g(x)
# computed according to g(x)
# 
def g_grad(b):
    g_left_x1 =  torch.pi**2*torch.sin(torch.pi*b[:,0,0]).reshape(-1,1,1) * torch.sin(torch.pi*b[:,1,0]).reshape(-1,1,1)
    g_left_x2 = -torch.pi**2*torch.cos(torch.pi*b[:,0,0]).reshape(-1,1,1) * torch.cos(torch.pi*b[:,1,0]).reshape(-1,1,1)
    g_left_grad = torch.cat((g_left_x1,g_left_x2),1)
   
    g_right_x1 = -torch.pi**2*torch.sin(torch.pi*b[:,0,1]).reshape(-1,1,1) * torch.sin(torch.pi*b[:,1,1]).reshape(-1,1,1)
    g_right_x2 =  torch.pi**2*torch.cos(torch.pi*b[:,0,1]).reshape(-1,1,1) * torch.cos(torch.pi*b[:,1,1]).reshape(-1,1,1)
    g_right_grad = torch.cat((g_right_x1,g_right_x2),1)

    g_bottom_x1 = -torch.pi**2*torch.cos(torch.pi*b[:,0,2]).reshape(-1,1,1) * torch.cos(torch.pi*b[:,1,2]).reshape(-1,1,1)
    g_bottom_x2 =  torch.pi**2*torch.sin(torch.pi*b[:,0,2]).reshape(-1,1,1) * torch.sin(torch.pi*b[:,1,2]).reshape(-1,1,1)
    g_bottom_grad = torch.cat((g_bottom_x1,g_bottom_x2),1)

    g_top_x1 =  torch.pi**2*torch.cos(torch.pi*b[:,0,3]).reshape(-1,1,1) * torch.cos(torch.pi*b[:,1,3]).reshape(-1,1,1)
    g_top_x2 = -torch.pi**2*torch.sin(torch.pi*b[:,0,3]).reshape(-1,1,1) * torch.sin(torch.pi*b[:,1,1]).reshape(-1,1,1)
    g_top_grad = torch.cat((g_top_x1,g_top_x2),1)

    return torch.cat((g_left_grad,g_right_grad,g_bottom_grad,g_top_grad),2)
