###### 二维例子
###### u=sin(pi*x)*sin(pi*y)    f=-laplace u + u = (2*pi^2+1)*u
from math import *
from pickle import NEXT_BUFFER

import numpy as np
import torch
import torch.nn as nn

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
        self.linear_out = torch.nn.Linear(layer_width, 1)
        self.acti = torch.sigmoid()

    def forward(self, x):
        y = self.linear_in(x) # fully connect layer
        y = y + acti(self.linear2(acti(self.linear1(y)))) # residual block 1
        y = y + acti(self.linear4(acti(self.linear3(y)))) # residual block 2
        y = y + acti(self.linear6(acti(self.linear5(y)))) # residual block 3
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

# 正态分布初始化参数
def initparam(model, sigma):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, sigma)
    return model

model = DeepNet(dimension, 6)
model = initparam(model, 0.5)
Net_f = DeepNet(2, 6)
Net_f = initparam(Net_f, 0.5)


# 方程右端源项f真解
def f(x):             
    f_x =(2*pi**2+1)*torch.sin(pi*x[:,0]).reshape(-1,1)*torch.sin(pi*x[:,1]).reshape(-1,1)
    return f_x

def f_whole(x):     # f = - \laplace u  +u f网络重建结果
    value = Net_f(x)
    return value

# exact solution  u
def u_ex(x):    #  u = sin(pi*x)*sin(pi*y)
    u_x1 = torch.sin(pi*x[:,0]).reshape(-1,1)
    u_x2 = torch.sin(pi*x[:,1]).reshape(-1,1)
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

    #  exact neumann boundary value 
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
   
    part_2_0 = torch.sum((g_hat_xb1 - g_xb1) ** 2) / xb1.size()[0] + torch.sum((g_hat_xb2 - g_xb2) ** 2) / xb2.size()[0]+ torch.sum((g_hat_xb3 - g_xb3) ** 2) / xb3.size()[0] +torch.sum((g_hat_xb4 - g_xb4) ** 2) / xb4.size()[0]
    part_2_1 = torch.sum((g_hat_xb1_x1 - g_xb1_x1) ** 2+(g_hat_xb1_x2 - g_xb1_x2) ** 2) / xb1.size()[0] + torch.sum((g_hat_xb2_x1 - g_xb2_x1) ** 2+(g_hat_xb2_x2 - g_xb2_x2) ** 2) / xb2.size()[0]+ torch.sum((g_hat_xb3_x1 - g_xb3_x1) ** 2+(g_hat_xb3_x2 - g_xb3_x2) ** 2) / xb3.size()[0] +torch.sum((g_hat_xb4_x1 - g_xb4_x1) ** 2+(g_hat_xb4_x2 - g_xb4_x2) ** 2) / xb4.size()[0]
    part_2 = part_2_0+part_2_1


#     # section3: in inverse problem, adding Dirichlet boundary condition
#     part_3 = torch.sum((model(xb1)-u_ex(xb1))**2)/xb1.size()[0]+torch.sum((model(xb2)-u_ex(xb2))**2)/xb2.size()[0]+torch.sum((model(xb3)-u_ex(xb3))**2)/xb3.size()[0]+torch.sum((model(xb4)-u_ex(xb4))**2)/xb4.size()[0]

    #  section 4  Data supervision Loss
    interior_domain = interior_data()          
    u_exact= u_ex(interior_domain)     
    part_4  = torch.sum( (model(interior_domain) - u_exact)**2) / interior_domain.size()[0]
    

    # Total loss
    lambda1 = 1 
    weight_data = 1    
    return 1*part_1 + lambda1 * part_2 / 1   + weight_data*part_4 # + lambda2*part_3/4    

import time

import torch.nn as nn
import torch.optim as optim

traintime = 20000
error_save_u = np.zeros(traintime)
error_save_f = np.zeros(traintime)
Loss = np.zeros(traintime)

#optimizer = optim.Adam(model.parameters())
optimizer = optim.Adam([{'params':model.parameters(),'lr':1e-3},{'params':Net_f.parameters(),'lr':1e-2}])
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # 每十次迭代，学习率减半


time_start = time.time()
torch.manual_seed(1000)
x = Gendata()
Nb = 100
xb1 = torch.rand(Nb, dimension)
xb1[:, 0] = torch.zeros(Nb)
xb2 = torch.rand(Nb, dimension)
xb2[:, 0] = torch.ones(Nb)
xb3 = torch.rand(Nb, dimension)
xb3[:, 1] = torch.zeros(Nb)
xb4 = torch.rand(Nb, dimension)
xb4[:, 1] = torch.ones(Nb)

xb1.requires_grad = True 
xb2.requires_grad = True
xb3.requires_grad = True
xb4.requires_grad = True

data_x = torch.cat([x,xb1],dim=0) 
data_x = torch.cat([data_x,xb2],dim=0) 
data_x = torch.cat([data_x,xb3],dim=0) 
data_x = torch.cat([data_x,xb4],dim=0) 

# # 方程右端源项f=f1*f2    令f1 = \sum{k=1...5} ak*sin(kx)
# # 给定ak  随机初值 a=[a1,a2,a3,a4,a5]   取值在（0，1）范围内
# alpha = torch.rand(5,1)

for i in range(traintime):
    optimizer.zero_grad()   
    x.requires_grad = True
    losses = DGM(x,xb1,xb2,xb3,xb4,data_x)
    losses.backward()
    optimizer.step()
    Loss[i] = float(losses)
    error_u =relative_error_u(x) 
    error_save_u[i] = float(error_u)
    error_f =relative_error_f(x)
    error_save_f[i] = float(error_f)
    
    if i % 50 == 0:
        print("current epoch is: ", i)
        print("current loss is: ", losses.detach())
        np.save("sin_H1_no_noise_loss.npy",Loss)
        print("current L2 error of u is: ", error_u.detach())
        np.save("sin_H1_no_noise_relative_error_u.npy", error_save_u)
        print("current L2 error of f is: ", error_f.detach())
        np.save("sin_H1_no_noise_relative_error_f.npy", error_save_f)
    if i == traintime - 1:
        #save entire model
        torch.save(model, 'sin_H1_no_noise_u_net.pth') 
        torch.save(Net_f, 'sin_H1_no_noise_f_net.pth')
        # save model parameters   
        torch.save(model.state_dict(), 'sin_H1_no_noise_u_net_params.pkl')
        torch.save(Net_f.state_dict(), 'sin_H1_no_noise_f_net_params.pkl')
        print("H1_net_u and net_f parameters saved")
time_end = time.time()
print('total time is: ', time_end-time_start, 'seconds') 
