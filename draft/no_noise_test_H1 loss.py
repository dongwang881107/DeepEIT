from math import *
from pickle import NEXT_BUFFER

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR

torch.set_default_tensor_type('torch.DoubleTensor') # 设置浮点类型为 torch.float64

# 定义激活函数: swish(x)
def acti(x):
    return x*torch.sigmoid(x)  

# 定义网络结构
class DeepNet(torch.nn.Module): 
    def __init__(self, input_width, layer_width):
        super(DeepNet, self).__init__()
        self.linear_in = torch.nn.Linear(input_width, layer_width)
        self.linear1 = torch.nn.Linear(layer_width, layer_width)
        self.linear2 = torch.nn.Linear(layer_width, layer_width)
        self.linear3 = torch.nn.Linear(layer_width, layer_width)
        self.linear4 = torch.nn.Linear(layer_width, layer_width)
        self.linear5 = torch.nn.Linear(layer_width, layer_width)
        self.linear6 = torch.nn.Linear(layer_width, layer_width)
#         self.linear7 = torch.nn.Linear(layer_width, layer_width)
#         self.linear8 = torch.nn.Linear(layer_width, layer_width)
#         self.linear9= torch.nn.Linear(layer_width, layer_width)
#         self.linear10= torch.nn.Linear(layer_width, layer_width)
        self.linear_out = torch.nn.Linear(layer_width, 1)

    def forward(self, x):
        y = self.linear_in(x) # fully connect layer
        y = y + acti(self.linear2(acti(self.linear1(y)))) # residual block 1
        y = y + acti(self.linear4(acti(self.linear3(y)))) # residual block 2
        y = y + acti(self.linear6(acti(self.linear5(y)))) # residual block 3
#         y = y + acti(self.linear8(acti(self.linear7(y))))  #residual block 3
#         y = y + acti(self.linear10(acti(self.linear9(y)))) # residual block 3
        output = self.linear_out(y) # fully connect layer
        return output
    
dimension = 2
Data_size = 1000
def Gendata():
    x = torch.rand(Data_size, dimension)
    return x

def interior_data():   #    取内部区域一小方块  1/16 面积  [0.5, 0.75]*[0.5, 0.75]
    x1= 0.5*torch.ones(1*Data_size,1)+0.25*torch.rand(1*Data_size,1)
    y1= 0.5*torch.ones(1*Data_size,1)+0.25*torch.rand(1*Data_size,1)
    x=torch.cat((x1,y1),1)
    return x

def interior_ring():
    Nb = 100
    x1 = 0.1*torch.rand(Nb, dimension)
    x1[:, 0] = 0.9*torch.zeros(Nb)
    x2 = 0.1+0.9*torch.rand(Nb, dimension)
    x2[:, 0] = 0.1*torch.ones(Nb)
    x3 = 0.9*torch.rand(Nb, dimension)
    x3[:, 1] = 0.9+0.1*torch.zeros(Nb)
    x4 = 0.9+0.1*torch.rand(Nb, dimension)
    x4[:, 1] =0.1+0.9* torch.ones(Nb)

    data_x = torch.cat([x1,x2],dim=0) 
    data_x = torch.cat([data_x,x3],dim=0) 
    data_x = torch.cat([data_x,xb4],dim=0) 
    return data_x


# 正态分布初始化参数
def initparam(model, sigma):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, sigma)
    return model

model = DeepNet(dimension, 6)
model = initparam(model, 0.5)
Net_f = DeepNet(1, 6)
Net_f = initparam(Net_f, 0.5)


# 方程右端源项f真解
def f(x):             

    f_x = -4*torch.exp(x[:,0]).reshape(-1,1)
    return f_x

def f_whole(x):     # f = - \laplace u  +u
    x1=x[:,0].reshape(-1,1)
    value = Net_f(x1)
    return value

# exact solution  u
def u_ex(x):    #  u = (x+y^2)*exp(x)

    u_x1 = (x[:,0]+x[:,1]**2).reshape(-1,1)
    u_x2 = torch.exp(x[:,0]).reshape(-1,1)
    u_x = u_x1*u_x2
    return u_x

def relative_error_u(x):
    predict = model(x)
    exact = u_ex(x)
    value = torch.sqrt(torch.sum((predict - exact )**2))/torch.sqrt(torch.sum((exact )**2))
    return value
def relative_error_f(x):
    predict = f_whole(x)
    exact = f(x)
    value = torch.sqrt(torch.sum((predict - exact )**2))/torch.sqrt(torch.sum((exact )**2))
    return value 
def L2_error_u(x):
    predict = model(x)
    exact = u_ex(x)
    value = torch.sqrt(torch.sum((predict - exact )**2))
    return value
def L2_error_f(x):
    predict = f_whole(x)
    exact = f(x)
    value = torch.sqrt(torch.sum((predict - exact )**2))
    return value 

 # Deep Galerkin Method
def DGM(x,xb1,xb2,xb3,xb4,data_x):
    
    # section 1  Equation   Loss
    u_hat = model(x)
    v = torch.ones_like(u_hat)
    ux = torch.autograd.grad(outputs=u_hat, inputs=x, grad_outputs=v, create_graph=True)[0]
    ux1 = ux[:, 0].reshape(-1, 1)     # transform the vector into a column vector
    ux2 = ux[:, 1].reshape(-1, 1)
    uxx1 = torch.autograd.grad(outputs=ux1, inputs=x, grad_outputs=v, create_graph=True)[0][:,0].reshape(-1,1)
    uxx2 = torch.autograd.grad(outputs=ux2, inputs=x, grad_outputs=v, create_graph=True)[0][:,1].reshape(-1,1)
    laplace_u = uxx1 + uxx2
    f_temp = f_whole(x)
    part_1 = torch.sum((-laplace_u + u_hat - f_temp) ** 2) / x.size()[0]

    #  section 2   Boundary Loss   (Neumann boundary condition)
#     #   First,  define boundary on the Rectangle domain
#     xb1 = torch.rand(num_boundary, dimension)
#     xb1[:, 0] = torch.zeros(num_boundary)        # xb1     left boundary
#     xb1.requires_grad = True
#     xb2 = torch.rand(num_boundary, dimension)
#     xb2[:, 0] = torch.ones(num_boundary)         # xb2     right boundary
#     xb2.requires_grad = True
#     xb3 = torch.rand(num_boundary, dimension)
#     xb3[:, 1] = torch.zeros(num_boundary)        # xb3     bottom boundary
#     xb3.requires_grad = True
#     xb4 = torch.rand(num_boundary, dimension)
#     xb4[:, 1] = torch.ones(num_boundary)         # xb4     top boundary
#     xb4.requires_grad = True


    #  exact neumann  boundary value
#     g_xb1 = -torch.exp(xb1[:, 0]).reshape(-1,1) * torch.exp(xb1[:,1]).reshape(-1,1)
#     g_xb2 = torch.exp(xb2[:, 0]).reshape(-1,1) * torch.exp(xb2[:,1]).reshape(-1,1)
#     g_xb3 = -torch.exp(xb3[:, 0]).reshape(-1,1) * torch.exp(xb3[:,1]).reshape(-1,1)
#     g_xb4 = torch.exp(xb4[:, 0]).reshape(-1,1) * torch.exp(xb4[:,1]).reshape(-1,1)
    
    g_xb1 = -(1+xb1[:,0]+xb1[:, 1]**2).reshape(-1,1)*torch.exp(xb1[:,0]).reshape(-1,1)
    g_xb2 =  (1+xb2[:,0]+xb2[:, 1]**2).reshape(-1,1)*torch.exp(xb2[:,0]).reshape(-1,1)
    g_xb3 = -2*(xb3[:, 1]).reshape(-1,1)*torch.exp(xb3[:,0]).reshape(-1,1)
    g_xb4 = 2*(xb4[:, 1]).reshape(-1,1)*torch.exp(xb4[:,0]).reshape(-1,1)

    
    g_xb1_x1 = -(2+xb1[:,0]+xb1[:, 1]**2).reshape(-1,1)*torch.exp(xb1[:,0]).reshape(-1,1)
    g_xb1_x2 = -2*(xb1[:, 1]).reshape(-1,1)*torch.exp(xb1[:,0]).reshape(-1,1)
   
    g_xb2_x1 = (2+xb2[:,0]+xb2[:, 1]**2).reshape(-1,1)*torch.exp(xb2[:,0]).reshape(-1,1)
    g_xb2_x2 = 2*(xb2[:, 1]).reshape(-1,1)*torch.exp(xb2[:,0]).reshape(-1,1)

    g_xb3_x1 = g_xb3
    g_xb3_x2 = -2*torch.exp(xb3[:,0]).reshape(-1,1)

    g_xb4_x1 = g_xb4
    g_xb4_x2 = 2*torch.exp(xb4[:,0]).reshape(-1,1)

    #  model neumann boundary value
    u_hat_xb1 = model(xb1)
    u_xb1_shape = torch.ones(u_hat_xb1.shape)
    u_xb1_x1 = torch.autograd.grad(outputs=u_hat_xb1, inputs=xb1, grad_outputs=u_xb1_shape, create_graph=True)[0][:, 0].reshape(-1, 1)
    g_hat_xb1 = -u_xb1_x1
    u_xb1_x1x1 = torch.autograd.grad(outputs=u_xb1_x1, inputs=xb1, grad_outputs=u_xb1_shape, create_graph=True)[0][:, 0].reshape(-1, 1)
    u_xb1_x1x2 = torch.autograd.grad(outputs=u_xb1_x1, inputs=xb1, grad_outputs=u_xb1_shape, create_graph=True)[0][:, 1].reshape(-1, 1)
    g_hat_xb1_x1 = - u_xb1_x1x1
    g_hat_xb1_x2 = - u_xb1_x1x2


    u_hat_xb2 = model(xb2)
    u_xb2_shape = torch.ones(u_hat_xb2.shape)
    u_xb2_x1 = torch.autograd.grad(outputs=u_hat_xb2, inputs=xb2, grad_outputs=u_xb2_shape, create_graph=True)[0][:, 0].reshape(-1, 1)
    g_hat_xb2 = u_xb2_x1
    u_xb2_x1x1 = torch.autograd.grad(outputs=u_xb2_x1, inputs=xb2, grad_outputs=u_xb2_shape, create_graph=True)[0][:, 0].reshape(-1, 1)
    u_xb2_x1x2 = torch.autograd.grad(outputs=u_xb2_x1, inputs=xb2, grad_outputs=u_xb2_shape, create_graph=True)[0][:, 1].reshape(-1, 1)
    g_hat_xb2_x1 = u_xb2_x1x1
    g_hat_xb2_x2 = u_xb2_x1x2



    u_hat_xb3 = model(xb3)
    u_xb3_shape = torch.ones(u_hat_xb3.shape)
    u_xb3_x2 = torch.autograd.grad(outputs=u_hat_xb3, inputs=xb3, grad_outputs=u_xb3_shape, create_graph=True)[0][:, 1].reshape(-1, 1)
    g_hat_xb3 = - u_xb3_x2
    u_xb3_x2x1 = torch.autograd.grad(outputs=u_xb3_x2, inputs=xb3, grad_outputs=u_xb3_shape, create_graph=True)[0][:, 0].reshape(-1, 1)
    u_xb3_x2x2 = torch.autograd.grad(outputs=u_xb3_x2, inputs=xb3, grad_outputs=u_xb3_shape, create_graph=True)[0][:, 1].reshape(-1, 1)
    g_hat_xb3_x1 = -u_xb3_x2x1
    g_hat_xb3_x2 = -u_xb3_x2x2

    u_hat_xb4 = model(xb4)
    u_xb4_shape = torch.ones(u_hat_xb4.shape)
    u_xb4_x2 = torch.autograd.grad(outputs=u_hat_xb4, inputs=xb4, grad_outputs=u_xb4_shape, create_graph=True)[0][:, 1].reshape(-1, 1)
    g_hat_xb4 = u_xb4_x2
    u_xb4_x2x1 = torch.autograd.grad(outputs=u_xb4_x2, inputs=xb4, grad_outputs=u_xb4_shape, create_graph=True)[0][:, 0].reshape(-1, 1)
    u_xb4_x2x2 = torch.autograd.grad(outputs=u_xb4_x2, inputs=xb4, grad_outputs=u_xb4_shape, create_graph=True)[0][:, 1].reshape(-1, 1)
    g_hat_xb4_x1 = u_xb4_x2x1
    g_hat_xb4_x2 = u_xb4_x2x2

#     boundary_loss = torch.sum((g_hat_xb1 - g_xb1) ** 2) / xb1.size()[0] + \
#                     torch.sum((g_hat_xb2 - g_xb2) ** 2) / xb2.size()[0] + \
#                     torch.sum((g_hat_xb3 - g_xb3) ** 2) / xb3.size()[0] + \
#                     torch.sum((g_hat_xb4 - g_xb4) ** 2) / xb4.size()[0]
   
    part_2_0 = torch.sum((g_hat_xb1 - g_xb1) ** 2) / xb1.size()[0] + torch.sum((g_hat_xb2 - g_xb2) ** 2) / xb2.size()[0]+ torch.sum((g_hat_xb3 - g_xb3) ** 2) / xb3.size()[0] +torch.sum((g_hat_xb4 - g_xb4) ** 2) / xb4.size()[0]
    part_2_1 = torch.sum((g_hat_xb1_x1 - g_xb1_x1) ** 2+(g_hat_xb1_x2 - g_xb1_x2) ** 2) / xb1.size()[0] + torch.sum((g_hat_xb2_x1 - g_xb2_x1) ** 2+(g_hat_xb2_x2 - g_xb2_x2) ** 2) / xb2.size()[0]+ torch.sum((g_hat_xb3_x1 - g_xb3_x1) ** 2+(g_hat_xb3_x2 - g_xb3_x2) ** 2) / xb3.size()[0] +torch.sum((g_hat_xb4_x1 - g_xb4_x1) ** 2+(g_hat_xb4_x2 - g_xb4_x2) ** 2) / xb4.size()[0]
    part_2 = part_2_0+part_2_1


#     # section3: in inverse problem, adding Dirichlet boundary condition
#     part_3 = torch.sum((model(xb1)-u_ex(xb1))**2)/xb1.size()[0]+torch.sum((model(xb2)-u_ex(xb2))**2)/xb2.size()[0]+torch.sum((model(xb3)-u_ex(xb3))**2)/xb3.size()[0]+torch.sum((model(xb4)-u_ex(xb4))**2)/xb4.size()[0]

    #  section 4  Data supervision Loss
    interior_domain = interior_data()          
    u_exact= u_ex(interior_domain)     
    part_4  = torch.sum( (model(interior_domain) - u_exact)**2) / interior_domain.size()[0]
    
    lambda1 = 1
    lambda2 = 100.0    
    weight_data = 1    
    return 1*part_1 + lambda1 * part_2 / 1   + weight_data*part_4 # + lambda2*part_3/4    

### load Net
model_u = torch.load('H1_no_noise_u_net.pth') 
model_f = torch.load('H1_no_noise_f_net.pth')

### test data
m,n = (500,500)
x = torch.linspace(0,1,m)
y = torch.linspace(0,1,n)
X,Y = torch.meshgrid(x,y)
x1 = X.reshape(m*n,1)
x2 = Y.reshape(m*n,1)
all_x = torch.cat([x1,x2],dim=1)

# print(x.shape)
# print(y.shape)
# print(x1.shape)
# print(x2.shape)
# print(all_x.shape)

u_net = model_u(all_x)
u_exact = u_ex(all_x)
np_u_net = u_net.detach().numpy() 
np_u_exact = u_exact.detach().numpy() 
np_all_x = all_x.detach().numpy() 

f_net = model_f(all_x[:,0].reshape(-1,1))
f_exact = f(all_x)
np_f_net = f_net.detach().numpy() 
np_f_exact = f_exact.detach().numpy() 

### save test data to files
np.save('H1_no noise_x.npy', np_all_x,allow_pickle=True, fix_imports=True)
np.save('H1_no noise_f_exact.npy',np_f_exact,allow_pickle=True, fix_imports=True)
np.save('H1_no noise_f_net.npy',np_f_net,allow_pickle=True, fix_imports=True)
np.save('H1_no noise_u_exact.npy',np_u_exact,allow_pickle=True, fix_imports=True)
np.save('H1_no noise_u_net.npy',np_u_net,allow_pickle=True, fix_imports=True)

print(11)
