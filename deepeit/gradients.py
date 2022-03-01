import torch
import numpy as np

# compute the gradient using torch.autograd.grad
# https://github.com/lululxvi/deepxde/blob/50b04a63e9d4f2647003924eebdac17609c65ffc/deepxde/gradients.py
# u = (u_1,...,u_N), x = (x_1,...,x_M)
# idx: the subindex of u to be computed
def jacobian(u, x, idx=[0]):
    idx = list(set(idx))
    idx.sort()
    assert((idx[0]>-1) & (idx[-1]<u.size()[1]))
    J = torch.zeros(u.size()[0], x.size()[1], len(idx))
    for i, id in enumerate(idx):
        J[:,:,i] = gradient(u[:,id].reshape(-1,1), x)
    return J

# compute the hessian matrix 
def hessian(u, x, idx=[0]):
    J = jacobian(u, x, idx)
    H = torch.zeros(u.size()[0], x.size()[1], x.size()[1], len(idx))
    for i in range(len(idx)):
        H[:,:,:,i] = jacobian(J[:,:,i], x, range(x.size()[1]))
    return H

# compute the gradient
def gradient(u, x):
    assert(u.size()[1] == 1)
    grad = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return grad

# compute the laplacian
def laplacian(u, x):
    grad = gradient(u, x)
    L = torch.zeros_like(u)
    if grad.requires_grad:
        for i in range(grad.size()[1]):
            L += gradient(grad[:,i].reshape(-1,1), x)[:,i].reshape(-1,1)
    return L

# compute the normal vector of points on the boundary
def boundary_normal(x, xmin, xmax):
    _n = -np.isclose(x.detach().numpy(), xmin).astype(float) + np.isclose(x.detach().numpy(), xmax)
    # For vertices, the normal is averaged for all directions
    idx = np.count_nonzero(_n, axis=-1) > 1
    if np.any(idx):
        l = np.linalg.norm(_n[idx], axis=-1, keepdims=True)
        _n[idx] /= l
    return torch.tensor(_n)