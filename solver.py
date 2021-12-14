import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time

from data import *

class Solver(object):
    def __init__(self, data_points, model_u, model_f, criterion, args):
        super().__init__()
        self.data_points = data_points
        self.model_u = model_u
        self.model_f = model_f

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_u.to(self.device)
        self.model_f.to(self.device)

        self.mode = args.mode
        self.num_interior_points = args.num_interior_points
        self.num_boundary_points = args.num_boundary_points
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.decay_iters = args.decay_iters
        self.print_iters = args.print_iters
        
        self.criterion = criterion
        self.optimizer = optim.Adam([{'params':self.model_u.parameters(),'lr':self.lr}, {'params':self.model_f.parameters(),'lr':self.lr}])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.decay_iters, gamma=0.5)
        
        self.save_path = args.save_path
        self.model_name = args.model_name
        self.args = args

    # forward propagate on the boundary points
    def propagate_boundary(self, b):
        ub_pred1 = self.model_u(b[:,:,0])
        ub_pred2 = self.model_u(b[:,:,1])
        ub_pred3 = self.model_u(b[:,:,2])
        ub_pred4 = self.model_u(b[:,:,3])

        return torch.cat((ub_pred1, ub_pred2, ub_pred3, ub_pred4), 1)

    # save model parameters
    def save_model(self):
        model_u_path = os.path.join(self.save_path, self.model_name+'_u.pkl')
        model_f_path = os.path.join(self.save_path, self.model_name+'_f.pkl')
        torch.save(self.model_u.state_dict(), model_u_path)
        torch.save(self.model_f.state_dict(), model_f_path)
        print('Model saved in {}'.format(model_u_path))
        print('Model saved in {}'.format(model_f_path))

    # load model parameters
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    # save training losses
    def save_loss(self, loss):
        loss_path = os.path.join(self.save_path, 'loss.npy')
        np.save(loss_path, loss)
        print('Loss saved in {}'.format(loss_path))

    # save training arguments
    def save_arg(self):
        arg_path = os.path.join(self.save_path, 'args.txt')
        with open(arg_path,'w') as file:
            for arg, value in self.args.items():
                file.writelines(arg+':'+str(value)+'\n')

    # train
    def train(self):
        start_time = time.time()
        training_loss = []
        print('Training start!')
        for epoch in range(self.num_epochs):
            # generate random training points inside the domain and on the boundary
            x = generate_interior_points(self.num_interior_points)
            b = generate_boundary_points(self.num_boundary_points)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward propagation
            ux_pred = self.model_u(x)
            ub_pred = self.propagate_boundary(b)
            us_pred = self.model_u(self.data_points)
            f_pred = self.model_f(x)
            # compute loss
            loss = self.criterion(x, b, self.data_points, ux_pred, ub_pred, us_pred, f_pred)
            # backward propagation
            loss.backward()
            # optimize
            self.optimizer.step()
            self.scheduler.step()
            # print statistics
            training_loss.append(loss.item())
            if epoch % self.print_iters == 0:
                print('epoch = {}, loss = {}, time = {}'.format(epoch, loss, time.time()-start_time))

        # save models
        self.save_model()
        # save losses
        self.save_loss(training_loss)
        # save arguments
        self.save_arg()
        print('Total running time is {}'.format(time.time()-start_time))
        print('Training finished!')
    
    # test
    def test(self):
        print('Done!')
            
