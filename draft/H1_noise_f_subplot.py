##### 将不同噪声水平下的 f_net 画在同一张图中

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

np_all_x = np.load('H1_no noise_x.npy')
np_f_exact = np.load('H1_no noise_f_exact.npy')
np_f_net_noise_1_1000 = np.load('H1_noise_1_1000_f_net.npy')
np_f_net_noise_1_100 = np.load('H1_noise_1_100_f_net.npy')
np_f_net_noise_3_100 = np.load('H1_noise_3_100_f_net.npy')
#np_f_net_noise_4_100 = np.load('H1_noise_4_100_f_net.npy')
np_f_net_noise_5_100 = np.load('H1_noise_5_100_f_net.npy')
np_f_net_noise_6_100 = np.load('H1_noise_6_100_f_net.npy')
np_f_net_no_noise= np.load('H1_no noise_f_net.npy')

m, n = (500,500)
X1 = np_all_x[:,0]
X1.shape = (m,n)
X2= np_all_x[:,1]
X2.shape = (m,n)
np_f_exact.shape=(m,n)
np_f_net_noise_1_1000.shape = (m,n)
np_f_net_noise_1_100.shape = (m,n)
np_f_net_noise_3_100.shape = (m,n)
#np_f_net_noise_4_100.shape = (m,n)
np_f_net_noise_5_100.shape = (m,n)
np_f_net_noise_6_100.shape = (m,n)

#####  不同噪声下的net_f 
# 第一种画图方式   每个子图都有一个colorbar
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,8))
im1 =ax[0,0].plot(X1,np_f_exact,color='b',marker="+")
ax[0,0].set_xlabel('$x_1$',fontsize=14)
ax[0,0].set_ylabel('f',fontsize=14)
ax[0,0].set_title('exact',fontsize=14)

im2=ax[0,1].plot(X1,np_f_net_noise_1_1000,color='b',marker="+")
ax[0,1].set_xlabel('$x_1$',fontsize=14)
ax[0,1].set_ylabel('$\\hat{f}^*$',fontsize=14)
ax[0,1].set_title('$\delta$=0.1%',fontsize=14)

im3=ax[1,0].plot(X1,np_f_net_noise_1_100,color='b',marker="+")
ax[1,0].set_xlabel('$x_1$',fontsize=14)
ax[1,0].set_ylabel('$\\hat{f}^*$',fontsize=14)
ax[1,0].set_title('$\delta$=1%',fontsize=14)

im4=ax[1,1].plot(X1,np_f_net_noise_5_100,color='b',marker="+")
ax[1,1].set_xlabel('$x_1$',fontsize=14)
ax[1,1].set_ylabel('$\\hat{f}^*$',fontsize=14)
ax[1,1].set_title('$\delta$=5%',fontsize=14)
# 子图间距调整
plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.savefig('f_different_noise_version0.png')

## 第二种画图方式，所有子图对应同一个colorbar
# fig, ax = plt.subplots(2, 2,figsize=(8,7))
# ax = ax.flatten()
# im =ax[0].contourf(X1,X2,np_u_exact, cmap='jet',vmin=0,vmax=6)
# ax[0].set_title('exact')
# im =ax[1].contourf(X1,X2,np_u_net_noise_1_1000, cmap='jet',vmin=0,vmax=6)
# ax[1].set_title('$\delta$=0.1%')
# im =ax[2].contourf(X1,X2,np_u_net_noise_1_100, cmap='jet',vmin=0,vmax=6)
# ax[2].set_title('$\delta$=1%')
# im =ax[3].contourf(X1,X2,np_u_net_noise_5_100, cmap='jet',vmin=0,vmax=6)
# ax[3].set_title('$\delta$=5%')
# # im =ax[4].contourf(X1,X2,np_u_net_noise_5_100, cmap='jet',vmin=0,vmax=6)
# # im =ax[5].contourf(X1,X2,np_u_net_noise_6_100, cmap='jet',vmin=0,vmax=6)
# fig.colorbar(im, ax=[ax[0], ax[1], ax[2], ax[3]], fraction=0.03, pad=0.05)
# plt.savefig('u_different_noise.png')

### 不同噪声下的 absolute error about f  |f_net-f_exact| 误差分布图
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
ax[0].plot(X1,abs(np_f_net_noise_1_1000-np_f_exact),color='b',marker="+")
ax[0].set_ylim([0,0.4])
ax[0].set_xlabel('$x_1$',fontsize=14)
ax[0].set_ylabel('$|\\hat{f}^* - f |$',fontsize=14)
ax[0].set_title('$\delta$=0.1%',fontsize=14)

ax[1].plot(X1,abs(np_f_net_noise_1_100-np_f_exact),color='b',marker="+")
ax[1].set_ylim([0,0.4])
ax[1].set_xlabel('$x_1$',fontsize=14)
ax[1].set_ylabel('$|\\hat{f}^* - f |$',fontsize=14)
ax[1].set_title('$\delta$=1%',fontsize=14)

ax[2].plot(X1,abs(np_f_net_noise_5_100-np_f_exact),color='b',marker="+")
ax[2].set_ylim([0,0.4])
ax[2].set_xlabel('$x_1$',fontsize=14)
ax[2].set_ylabel('$|\\hat{f}^* - f |$',fontsize=14)
ax[2].set_title('$\delta$=5%',fontsize=14)
# 子图间距调整
plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.savefig('absolute_error_f_different_noise_version0.png')

# # 设置同一个colorbar
# # 前面三个子图的总宽度 为 全部宽度的 0.9；剩下的0.1用来放置colorbar
# fig.subplots_adjust(right=0.9)
# #colorbar 左 下 宽 高 
# l = 0.92
# b = 0.12
# w = 0.015
# h = 0.76
# #对应 l,b,w,h；设置colorbar位置；
# rect = [l,b,w,h] 
# cbar_ax = fig.add_axes(rect) 
# plt.colorbar(im, cax=cbar_ax)
# plt.savefig('absolute_error_f_different_noise_version0.png')
# plt.show()


##############################
##############################
#### 大噪声下的返源问题图像解 Net_f  
np_f_net_noise_1_10 = np.load('H1_noise_1_10_f_net.npy')
np_f_net_noise_15_100 = np.load('H1_noise_15_100_f_net.npy')
np_f_net_noise_1_10.shape = (m,n)
np_f_net_noise_15_100.shape = (m,n)
# Net_f
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
im1 =ax[0].plot(X1,np_f_net_noise_1_10,color='b',marker="+")
ax[0].set_title('$\delta$=10%')
im2=ax[1].plot(X1,np_f_net_noise_15_100,color='b',marker="+")
ax[1].set_title('$\delta$=15%')
# absolute error_f
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
im1 =ax[0].plot(X1,abs(np_f_net_noise_1_10-np_f_exact), color='b',marker="+")
ax[0].set_title('$\delta$=10%')
im2=ax[1].plot(X1,abs(np_f_net_noise_15_100-np_f_exact), color='b',marker="+")
ax[1].set_title('$\delta$=15%')

