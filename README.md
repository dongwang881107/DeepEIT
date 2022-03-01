# DeepEIT: Recovery of the conductivity in 3-dimensional medium using deep neural network approximations
DeepEIT is an open-source Python-based code to solve an inverse problem for an elliptic equation and determine the conductivity using deep neural networks. Specifically, the proposed algorithm projects this problem into a finite dimensional space by approximating both the unknown conductivity and the corresponding solution using two independent neural networks, which are jointly trained by minimizing a novel loss function. 

## Method
Consider the following elliptic equation with Neumann boundary condition

<img src="https://latex.codecogs.com/svg.image?\begin{cases}-\nabla\cdot(\sigma\nabla&space;u)=f,&space;x\in&space;\Omega\\\sigma&space;\frac{\partial&space;u}{\partial&space;\nu}=g,&space;x\in\partial\Omega\\u=u_0,&space;x\in\Omega_0\end{cases}" title="\begin{cases}-\nabla\cdot(\sigma\nabla u)=f, x\in \Omega\\\sigma \frac{\partial u}{\partial \nu}=g, x\in\partial\Omega\\u=u_0, x\in\Omega_0\end{cases}" /> 

with <img src="https://latex.codecogs.com/svg.image?\inline&space;\Omega_0\subset\Omega\subset\mathbb{R}^2" title="\inline \Omega_0\subset\Omega\subset\mathbb{R}^2" />. In order to solve the inverse problem, we first construct two deep neural networks, one appximates the solution and the other approximates the conductivity. Then we propose a novel loss function to replace the Neumann boundary control with interior control by introducing a predifined harmonic function G.

<img src="https://latex.codecogs.com/svg.image?\inline&space;\begin{align*}J(\theta_1,\theta_2)=&space;&\|-\nabla\cdot(\tilde\sigma(\theta_1,\cdot)\nabla\tilde&space;u(\theta_2,\cdot))-f\|^2_{L^{2}(\Omega)}&plus;\\&\lambda_{11}\|\Delta(\tilde&space;u(\theta_2,\cdot)\|^2_{L^2(\Omega)}&plus;\lambda_{12}\|\nabla(\tilde&space;u(\theta_2,\cdot)-G)\|^2_{L^2(\Omega)}&plus;\lambda_{13}\|\tilde&space;u(\theta_2,\cdot)-G\|^2_{L^2(\Omega)}&plus;\\&\lambda_2\|\tilde&space;u(\theta_2,\cdot)-u_0\|^2_{L^2(\Omega_0)}&plus;\\&\lambda_3\|\tilde\sigma(\theta_1,\cdot)-\sigma_b\|^2_{L^2(\partial\Omega)}\end{align*}" title="\inline \begin{align*}J(\theta_1,\theta_2)= &\|-\nabla\cdot(\tilde\sigma(\theta_1,\cdot)\nabla\tilde u(\theta_2,\cdot))-f\|^2_{L^{2}(\Omega)}+\\&\lambda_{11}\|\Delta(\tilde u(\theta_2,\cdot)\|^2_{L^2(\Omega)}+\lambda_{12}\|\nabla(\tilde u(\theta_2,\cdot)-G)\|^2_{L^2(\Omega)}+\lambda_{13}\|\tilde u(\theta_2,\cdot)-G\|^2_{L^2(\Omega)}+\\&\lambda_2\|\tilde u(\theta_2,\cdot)-u_0\|^2_{L^2(\Omega_0)}+\\&\lambda_3\|\tilde\sigma(\theta_1,\cdot)-\sigma_b\|^2_{L^2(\partial\Omega)}\end{align*}" />

Finally we jointly train the nerual networks and reconstruct both the solution and the conductivity.

## Result
We perform the proposed algorithm on 3-dimensional medium, and the results are shown below.
- Hyper-parameters and reconstruction errors on different noise levels

<img width="808" alt="image" src="https://user-images.githubusercontent.com/7763153/156134531-7fa8fc74-3ae2-4a34-842d-544e8b3fa0f0.png">

- x=100
![3d-x](https://user-images.githubusercontent.com/7763153/156133608-6736dd5e-9d42-4ba5-bfd2-52ced395c4c8.png)
- y=100
![3d-y](https://user-images.githubusercontent.com/7763153/156133648-5d356fc2-14df-4f43-9ea6-37ddd31e9b30.png)
- z=100
![3d-z](https://user-images.githubusercontent.com/7763153/156133657-6813d151-a223-4e2a-9127-85150da4d849.png)

## Dependencies
The code has been tested on 
- macOS Big Sur & Python 3.8.8 & PyTorch 1.10.0
- Ubuntu 18.04.5 & 3.8.11 & PyTorch 1.0.0+cu111

## Usage
```
usage:
    python main.py {train, test, plot} [optional arguments]

    positional arguments:
      {train,test,plot}     train | test | plot
        train               training mode
        test                testing mode
        plot                plotting mode

    optional arguments:
      -h, --help            show this help message and exit
      --device DEVICE       cpu | cuda
      --seed SEED           random seed
      --num_channels        hidden layer width of network
      --num_blocks          number of residual blocks of network
      --acti ACTI           activation function of the network
      --dim DIM             dimension of space
      --xmin XMIN           lower bound of Omega
      --xmax XMAX           upper bound of Omega
      --csv_path            path of csv path to store training parameteres
      --save_path           saved path of the results
```

## Funding Sources
This work is supported by NSFC (No.11971104, 11531005) and National Key R&D Program of China (2020YFA0713800).

