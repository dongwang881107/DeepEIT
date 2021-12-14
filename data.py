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

# generate supervised points inside the domain Omega_0
# square domain [lower,upper]x[lower,upper]
def generate_supervised_points(num_points, lower, upper, dim=2):   
    return lower+(upper-lower)*torch.rand(num_points, dim)

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
def g_neumann(b):
    g_left   = -torch.pi*torch.cos(torch.pi*b[:,0,0]).reshape(-1,1) * torch.sin(torch.pi*b[:,1,0]).reshape(-1,1)
    g_right  =  torch.pi*torch.cos(torch.pi*b[:,0,1]).reshape(-1,1) * torch.sin(torch.pi*b[:,1,1]).reshape(-1,1)
    g_bottom = -torch.pi*torch.sin(torch.pi*b[:,0,2]).reshape(-1,1) * torch.cos(torch.pi*b[:,1,2]).reshape(-1,1)
    g_top    =  torch.pi*torch.sin(torch.pi*b[:,0,3]).reshape(-1,1) * torch.cos(torch.pi*b[:,1,3]).reshape(-1,1)
    return torch.cat((g_left,g_right,g_bottom,g_top),1)

# gradient of ground truth Neumann boundary condition g(x)
# computed according to g(x)
def g_neumann_grad(b):
    g_left_grad_x1 =  torch.pi**2*torch.sin(torch.pi*b[:,0,0]).reshape(-1,1,1) * torch.sin(torch.pi*b[:,1,0]).reshape(-1,1,1)
    g_left_grad_x2 = -torch.pi**2*torch.cos(torch.pi*b[:,0,0]).reshape(-1,1,1) * torch.cos(torch.pi*b[:,1,0]).reshape(-1,1,1)
    g_left_grad = torch.cat((g_left_grad_x1,g_left_grad_x2),1)
   
    g_right_grad_x1 = -torch.pi**2*torch.sin(torch.pi*b[:,0,1]).reshape(-1,1,1) * torch.sin(torch.pi*b[:,1,1]).reshape(-1,1,1)
    g_right_grad_x2 =  torch.pi**2*torch.cos(torch.pi*b[:,0,1]).reshape(-1,1,1) * torch.cos(torch.pi*b[:,1,1]).reshape(-1,1,1)
    g_right_grad = torch.cat((g_right_grad_x1,g_right_grad_x2),1)

    g_bottom_grad_x1 = -torch.pi**2*torch.cos(torch.pi*b[:,0,2]).reshape(-1,1,1) * torch.cos(torch.pi*b[:,1,2]).reshape(-1,1,1)
    g_bottom_grad_x2 =  torch.pi**2*torch.sin(torch.pi*b[:,0,2]).reshape(-1,1,1) * torch.sin(torch.pi*b[:,1,2]).reshape(-1,1,1)
    g_bottom_grad = torch.cat((g_bottom_grad_x1,g_bottom_grad_x2),1)

    g_top_grad_x1 =  torch.pi**2*torch.cos(torch.pi*b[:,0,3]).reshape(-1,1,1) * torch.cos(torch.pi*b[:,1,3]).reshape(-1,1,1)
    g_top_grad_x2 = -torch.pi**2*torch.sin(torch.pi*b[:,0,3]).reshape(-1,1,1) * torch.sin(torch.pi*b[:,1,1]).reshape(-1,1,1)
    g_top_grad = torch.cat((g_top_grad_x1,g_top_grad_x2),1)

    return torch.cat((g_left_grad,g_right_grad,g_bottom_grad,g_top_grad),2)

# compute loss on the interior points (equation)
# use torch.autograd.grad to compute the gradients
def interior_loss(x, ux_pred, f_pred):
    tmp = torch.ones_like(ux_pred)
    grad_u = torch.autograd.grad(outputs=ux_pred, inputs=x, grad_outputs=tmp, create_graph=True)[0]
    grad_u_x1 = grad_u[:,0].reshape(-1,1)
    grad_u_x2 = grad_u[:,1].reshape(-1,1)
    lap_u1 = torch.autograd.grad(ouputs=grad_u_x1, input=x, grad_outputs=tmp, create_graph=True)[0][:,0].reshape(-1,1)
    lap_u2 = torch.autograd.grad(ouputs=grad_u_x2, input=x, grad_outputs=tmp, create_graph=True)[0][:,1].reshape(-1,1)
    lap_u = lap_u1 + lap_u2

    return torch.sum((-lap_u+ux_pred-f_pred)**2)/x.size()[0]

# compute loss on the boundary points (boundary)
# use torch.autograd.grad to compute the gradients
def boundary_loss(b, ub_pred):
    tmp = torch.ones_like(ub_pred[:,0])
    grad_u_left = torch.autograd.grad(outputs=ub_pred[:,0], inputs=b[:,:,0], grad_outputs=tmp, create_graph=True)[0]
    g_left_pred = -grad_u_left[:,0].reshape(-1,1)
    g_left_grad_pred_x1 = torch.autograd.grad(outputs=g_left_pred, inputs=b[:,:,0], grad_outputs=tmp, create_graph=True)[0][:,0].reshape(-1,1)
    g_left_grad_pred_x2 = torch.autograd.grad(outputs=g_left_pred, inputs=b[:,:,0], grad_outputs=tmp, create_graph=True)[0][:,1].reshape(-1,1)
    g_left_grad_pred = torch.cat((g_left_grad_pred_x1,g_left_grad_pred_x2),1)

    tmp = torch.ones_like(ub_pred[:,1])
    grad_u_right = torch.autograd.grad(outputs=ub_pred[:,1], inputs=b[:,:,1], grad_outputs=tmp, create_graph=True)[0]
    g_right_pred = grad_u_right[:,0].reshape(-1,1)
    g_right_grad_pred_x1 = torch.autograd.grad(outputs=g_right_pred, inputs=b[:,:,0], grad_outputs=tmp, create_graph=True)[0][:,0].reshape(-1,1)
    g_right_grad_pred_x2 = torch.autograd.grad(outputs=g_right_pred, inputs=b[:,:,0], grad_outputs=tmp, create_graph=True)[0][:,1].reshape(-1,1)
    g_right_grad_pred = torch.cat((g_right_grad_pred_x1,g_right_grad_pred_x2),1)

    tmp = torch.ones_like(ub_pred[:,2])
    grad_u_bottom = torch.autograd.grad(outputs=ub_pred[:,2], inputs=b[:,:,2], grad_outputs=tmp, create_graph=True)[0]
    g_bottom_pred = -grad_u_bottom[:,0].reshape(-1,1)
    g_bottom_grad_pred_x1 = torch.autograd.grad(outputs=g_bottom_pred, inputs=b[:,:,2], grad_outputs=tmp, create_graph=True)[0][:,0].reshape(-1,1)
    g_bottom_grad_pred_x2 = torch.autograd.grad(outputs=g_bottom_pred, inputs=b[:,:,2], grad_outputs=tmp, create_graph=True)[0][:,1].reshape(-1,1)
    g_bottom_grad_pred = torch.cat((g_bottom_grad_pred_x1,g_bottom_grad_pred_x2),1)

    tmp = torch.ones_like(ub_pred[:,3])
    grad_u_top = torch.autograd.grad(outputs=ub_pred[:,3], inputs=b[:,:,3], grad_outputs=tmp, create_graph=True)[0]
    g_top_pred = grad_u_top[:,0].reshape(-1,1)
    g_top_grad_pred_x1 = torch.autograd.grad(outputs=g_top_pred, inputs=b[:,:,3], grad_outputs=tmp, create_graph=True)[0][:,0].reshape(-1,1)
    g_top_grad_pred_x2 = torch.autograd.grad(outputs=g_top_pred, inputs=b[:,:,3], grad_outputs=tmp, create_graph=True)[0][:,1].reshape(-1,1)
    g_top_grad_pred = torch.cat((g_top_grad_pred_x1,g_top_grad_pred_x2),1)

    g_pred = torch.cat((g_left_pred, g_right_pred, g_bottom_pred, g_top_pred),1)
    g_grad_pred = torch.cat((g_left_grad_pred, g_right_grad_pred, g_bottom_grad_pred, g_top_grad_pred),2)

    g_exact = g_neumann(b)
    g_grad_exact = g_neumann_grad(b)

    return torch.sum((g_pred-g_exact)**2+(g_grad_pred-g_grad_exact)**2)/b.size()[0]/b.size()[2]

# compute loss on the supervised points (supervised)
def supervised_loss(s, us_pred):
    us_exact = u(s)
    return torch.sum((us_pred-us_exact)**2)/s.size()[0]

# combine all the losses together
def compute_loss(x, b, s, ux_pred, ub_pred, us_pred, f_pred, args):
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    return  interior_loss(x, ux_pred, f_pred) + lambda1*boundary_loss(b, ub_pred) + lambda2*supervised_loss(s, us_pred)