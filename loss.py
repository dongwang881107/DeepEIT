import torch
from data import *

# compute loss on the interior points (equation)
# use torch.autograd.grad to compute the gradients
def interior_loss(x, ux_pred, f_pred):
    tmp = torch.ones_like(ux_pred)
    grad_u = torch.autograd.grad(outputs=ux_pred, inputs=x, grad_outputs=tmp, create_graph=True)[0]
    grad_u_x1 = grad_u[:,0].reshape(-1,1)
    grad_u_x2 = grad_u[:,1].reshape(-1,1)
    lap_u1 = torch.autograd.grad(outputs=grad_u_x1, inputs=x, grad_outputs=tmp, create_graph=True)[0][:,0].reshape(-1,1)
    lap_u2 = torch.autograd.grad(outputs=grad_u_x2, inputs=x, grad_outputs=tmp, create_graph=True)[0][:,1].reshape(-1,1)
    lap_u = lap_u1 + lap_u2

    return torch.sum((-lap_u+ux_pred-f_pred)**2)/x.size()[0]

# compute loss on one side of the boundary
# square boundary condition
# use torch.autograd.grad to compute the gradients
def side_loss(b, ub_pred, norm_vcetor):
    tmp = torch.ones_like(ub_pred)
    grad_u = torch.autograd.grad(outputs=ub_pred, inputs=b, grad_outputs=tmp, create_graph=True)[0]
    g_pred = (norm_vcetor[0]*grad_u[:,0]+norm_vcetor[1]*grad_u[:,1]).reshape(-1,1)
    g_grad_pred_x1 = torch.autograd.grad(outputs=g_pred, inputs=b, grad_outputs=tmp, create_graph=True)[0][:,0].reshape(-1,1,1)
    g_grad_pred_x2 = torch.autograd.grad(outputs=g_pred, inputs=b, grad_outputs=tmp, create_graph=True)[0][:,1].reshape(-1,1,1)
    g_grad_pred = torch.cat((g_grad_pred_x1,g_grad_pred_x2),1)

    return g_pred, g_grad_pred

# compute loss on the boundary points (boundary)
# combine side losses together
def boundary_loss(b_left, b_right, b_bottom, b_top, ub_left_pred, ub_right_pred, ub_bottom_pred, ub_top_pred):
    g_left_pred, g_left_grad_pred = side_loss(b_left, ub_left_pred, [-1,0])
    g_right_pred, g_right_grad_pred = side_loss(b_right, ub_right_pred, [1,0])
    g_bottom_pred, g_bottom_grad_pred = side_loss(b_bottom, ub_bottom_pred, [0,-1])
    g_top_pred, g_top_grad_pred = side_loss(b_top, ub_top_pred, [0,1])

    g_pred = torch.cat((g_left_pred, g_right_pred, g_bottom_pred, g_top_pred),1)
    g_grad_pred = torch.cat((g_left_grad_pred, g_right_grad_pred, g_bottom_grad_pred, g_top_grad_pred),2)

    g_exact = g_neumann(b_left, b_right, b_bottom, b_top)
    g_grad_exact = g_neumann_grad(b_left, b_right, b_bottom, b_top)

    return torch.sum((g_pred-g_exact)**2)/b_left.size()[0] + torch.sum((g_grad_pred-g_grad_exact)**2)/b_left.size()[0]/4

# compute loss on the supervised points (supervised)
def supervised_loss(s, us_pred):
    us_exact = u(s)
    return torch.sum((us_pred-us_exact)**2)/s.size()[0]

# combine all the losses together
def compute_loss(x, b_left, b_right, b_bottom, b_top, s, ux_pred, ub_left_pred, ub_right_pred, ub_bottom_pred, ub_top_pred, us_pred, f_pred, args):
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    return  interior_loss(x, ux_pred, f_pred) + lambda1*boundary_loss(b_left, b_right, b_bottom, b_top, ub_left_pred, ub_right_pred, ub_bottom_pred, ub_top_pred) + lambda2*supervised_loss(s, us_pred)