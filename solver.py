import torch
import time
import torch.optim as optim

from data import *
from model import *
from loss import *
from metric import *
from postprocessing import *

class Solver(object):
    def __init__(self, dataset, args):
        super().__init__()

        # load shared parameters
        self.save_path = args.save_path
        self.model_name = args.model_name
        self.device = torch.device(args.device)
        self.args = args

        # generate networks and move to self.device
        self.model_u = ResNet(args.num_channels, args.num_blocks, args.acti).to(self.device)
        self.model_f = ResNet(args.num_channels, args.num_blocks, args.acti).to(self.device)
        
        # move supervised_points to self.device
        self.s = dataset.points.to(self.device)

        # train | test
        if args.mode == 'train':
            # loss function and optimizer
            self.criterion = compute_loss
            self.optimizer = optim.Adam([{'params':self.model_u.parameters(),'lr':args.lr_u}, {'params':self.model_f.parameters(),'lr':args.lr_f}])
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_iters, gamma=args.gamma)
            # load training parameters
            self.num_interior_points = args.num_interior_points
            self.num_boundary_points = args.num_boundary_points
            self.num_epochs = args.num_epochs
            self.print_iters = args.print_iters
            self.loss_name = args.loss_name
            self.arg_name = args.arg_name
        elif args.mode == 'test':
            #load testing parameters
            self.testing_result_name = args.testing_result_name
            # load model
            load_model(self.model_u, self.model_f, args)

    # train
    def train(self):
        start_time = time.time()
        training_loss = []
        us_exact = u(self.s).to(self.device)

        print('Training start!')
        for epoch in range(self.num_epochs):
            # generate random training points inside the domain and on the boundary
            x = generate_interior_points(self.num_interior_points)
            b_left, b_right, b_bottom, b_top = generate_boundary_points(self.num_boundary_points)
            
            x.requires_grad = True
            b_left.requires_grad = True
            b_right.requires_grad = True
            b_bottom.requires_grad = True
            b_top.requires_grad = True

            x = x.to(self.device)
            b_left = b_left.to(self.device)
            b_right = b_right.to(self.device)
            b_bottom = b_bottom.to(self.device)
            b_top = b_top.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward propagation
            ux_pred = self.model_u(x)
            ub_left_pred = self.model_u(b_left)
            ub_right_pred = self.model_u(b_right)
            ub_bottom_pred = self.model_u(b_bottom)
            ub_top_pred = self.model_u(b_top)
            us_pred = self.model_u(self.s)
            f_pred = self.model_f(x)
            # compute loss
            loss = self.criterion(x, b_left, b_right, b_bottom, b_top, self.s, ux_pred, ub_left_pred, ub_right_pred, ub_bottom_pred, ub_top_pred, us_pred, us_exact, f_pred, self.args)
            # backward propagation
            loss.backward()
            # optimize
            self.optimizer.step()
            self.scheduler.step()
            # print statistics
            training_loss.append(loss.item())
            if epoch % self.print_iters == 0:
                print('epoch = {}, loss = {:.4f}, time = {:.4f}'.format(epoch, loss.detach(), time.time()-start_time))

        # save models
        save_model(self.model_u, self.model_f, self.args)
        # save losses
        save_loss(training_loss, self.args)
        # save arguments
        save_arg(self.args)
        print('Total training time is {:.4f} s'.format(time.time()-start_time))
        print('Training finished!')
    
    # test
    def test(self):
        start_time = time.time()

        print('Testing start!')
        with torch.no_grad():
            u_pred = self.model_u(self.s)
            f_pred = self.model_f(self.s)
            u_exact = u(self.s).to(self.device)
            f_exact = f(self.s).to(self.device)
            u_relative_error, u_l2_error = compute_metric(u_pred, u_exact) 
            f_relative_error, f_l2_error = compute_metric(f_pred, f_exact) 

        # print results
        print('Relative error of u is {:.4f}'.format(u_relative_error))
        print('L2 error of u is {:.4f}'.format(u_l2_error))
        print('Relative error of f is {:.4f}'.format(f_relative_error))
        print('L2 error of f is {:.4f}'.format(f_l2_error))

        # save results
        save_result(u_relative_error, f_relative_error, u_l2_error, f_l2_error, u_pred, u_exact, f_pred, f_exact, self.args)
        print('Total testing time is {:.4f} s'.format(time.time()-start_time))
        print('Testing finished!')
