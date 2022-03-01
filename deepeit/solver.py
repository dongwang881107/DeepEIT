import torch
import time
import torch.optim as optim

from deepeit.model import *
from deepeit.metric import *
from deepeit.data import *
from deepeit.pde import *
from deepeit.postprocessing import *

class Solver(object):
    def __init__(self, model_u, model_s, loss_func, metric_func, pde, args):
        super().__init__()

        # load shared parameters
        self.save_path = args.save_path
        self.dim = args.dim
        self.xmin = args.xmin
        self.xmax = args.xmax
        self.device = torch.device(args.device)
        self.args = args
        self.metric_func = metric_func
        self.pde = pde

        # move networks to self.device
        self.model_u = model_u.to(self.device)
        self.model_s = model_s.to(self.device)

        # train | test
        if args.mode == 'train':
            self.std = args.std
            self.omin = args.omin
            self.omax = args.omax
            # loss function
            self.loss_func = loss_func
            # load training parameters
            self.num_interior_points = args.num_interior_points
            self.num_boundary_points = args.num_boundary_points
            self.num_observed_points = args.num_observed_points
            self.lr_u = args.lr_u
            self.lr_s = args.lr_s
            self.decay_iters = args.decay_iters
            self.gamma = args.gamma
            self.num_epochs = args.num_epochs
            self.print_iters = args.print_iters
            self.save_iters = args.save_iters
            self.loss_name = args.loss_name
            self.arg_name = args.arg_name
            self.scheduler = args.scheduler
            # load validation parameters
            self.num_validation_points = args.num_validation_points
        elif args.mode == 'test':            
            #load testing parameters
            self.num_testing_points = args.num_testing_points
            self.pred_name = args.pred_name

    # train
    def train(self):
        start_time = time.time()

        # generate points and move to device
        data_x = Data(self.num_interior_points, self.xmin, self.xmax, self.dim) # interior points
        data_b = Data(self.num_boundary_points, self.xmin, self.xmax, self.dim) # boundary points
        data_o = Data(self.num_observed_points, self.omin, self.omax, self.dim) # observed points
        data_v = Data(self.num_validation_points, self.xmin, self.xmax, self.dim) # validation points
        
        # add noise to observed data
        o = data_o.random_interior_points().to(self.device)
        noisy_u = self.pde.solution(o)
        if not np.isclose(self.std, 0):
            std_u = torch.std(noisy_u)
            noisy_u = noisy_u+ self.std*std_u*torch.randn(noisy_u.size()).to(self.device)
            print('{}% percent noise add'.format(self.std*100))

        # set up optimizer and scheduler
        optimizer, scheduler = set_optim(self.model_u, self.model_s, self.args)

        # load checkpoint and statistics
        print('{:-^115s}'.format('Training start!'))
        start_epoch = load_checkpoint(self.model_u, self.model_s, optimizer, self.args)
        training_loss_total, training_loss_terms, metric_u, metric_s, lr = load_statistics(start_epoch, self.args)
        
        if self.num_epochs !=0:
            for epoch in range(start_epoch, self.num_epochs):
                # generate random training points inside the domain and on the boundary
                x = data_x.random_interior_points().to(self.device)
                b = data_b.random_boundary_points().to(self.device)
                v = data_v.uniform_interior_points().to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # compute loss
                loss_total, loss_terms = self.loss_func(x, b, o, noisy_u, self.args)
                # backward propagation
                loss_total.backward(retain_graph=True)
                # save statistics
                training_loss_total.append(loss_total.item())
                training_loss_terms.append(loss_terms)
                if self.scheduler == 'reduce':
                    lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
                else:
                    lr.append(scheduler.get_lr()[0])
                # do validation
                with torch.no_grad():
                    pred_u = self.model_u(v)
                    pred_s = self.model_s(v)
                    exact_u = self.pde.solution(v).to(self.device)
                    exact_s = self.pde.inverse(v).to(self.device)
                    metric_u.append(self.metric_func(pred_u, exact_u))
                    metric_s.append(self.metric_func(pred_s, exact_s))
                # optimize
                optimizer.step()
                if self.scheduler == 'reduce':
                    scheduler.step(metric_s[-1][list(metric_s[-1].keys())[0]])
                else:
                    scheduler.step()

                # print statistics
                if (epoch+1) % self.print_iters == 0:
                    print_stat(epoch, training_loss_total, metric_u, metric_s, start_time)

                # save intermediate models
                if (epoch+1) % self.save_iters == 0:
                    save_checkpoint(self.model_u, self.model_s, optimizer, epoch, self.args)
                    save_loss(training_loss_total, training_loss_terms, self.args)
                    save_metric(metric_u, metric_s, self.args)
                    save_lr(lr, self.args)
                    save_arg(self.args)

            print('{:-^115s}'.format('Training finished!'))
            print('Total training time is {:.4f} s'.format(time.time()-start_time))
            print('{:-^115s}'.format('Saving results!'))
            # save losses
            save_loss(training_loss_total, training_loss_terms, self.args)
            # save errors
            save_metric(metric_u, metric_s, self.args)
            # save learning rates
            save_lr(lr, self.args)

        print('{:-^115s}'.format('Saving results!'))
        # save models
        save_model(self.model_u, self.model_s, self.args)
        # save arguments
        save_arg(self.args)
        print('{:-^115s}'.format('Done!'))
    
    # test
    def test(self):
        start_time = time.time()

        # generate points and move to device
        data_t = Data(self.num_testing_points, self.xmin, self.xmax, self.dim)
        t = data_t.uniform_interior_points().to(self.device)

        # load model
        load_model(self.model_u, self.model_s, self.args)

        print('{:-^115s}'.format('Testing start!'))
        with torch.no_grad():
            pred_u = self.model_u(t)
            pred_s = self.model_s(t)
            exact_u = self.pde.solution(t).to(self.device)
            exact_s = self.pde.inverse(t).to(self.device)
            error_u = self.metric_func(pred_u, exact_u) 
            error_s = self.metric_func(pred_s, exact_s) 

        print_error(error_u, error_s)
        print('{:-^115s}'.format('Testing finished!'))
        print('Total testing time is {:.4f} s'.format(time.time()-start_time))

        # save results
        if self.args.not_save_result:
            print('{:-^115s}'.format('Saving results!'))
            save_pred(pred_u.cpu(), exact_u.cpu(), pred_s.cpu(), exact_s.cpu(), self.args)
            save_error(error_u, error_s, self.args)
            print('{:-^115s}'.format('Done!'))
        
