import os
import time
import argparse
import warnings

from deepeit.solver import *
from deepeit.data import *
from deepeit.postprocessing import *
from deepeit.loss import *
from deepeit.pde import *

def main(args):
    # set seed and data type
    torch.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.DoubleTensor')
    warnings.filterwarnings('ignore')

    # determine equation
    def equation(x, u, s):
        grad_u = gradient(u,x)
        grad_s = gradient(s,x)
        lap_u = laplacian(u,x)
        return torch.sum(grad_u*grad_s,1).reshape(-1,1) + s*lap_u

    # determine solution
    def g_func(x):
        return (torch.sin(np.pi*x[:,0])*torch.sinh(np.pi*x[:,1])*x[:,2]).reshape(-1,1) # 2D Case
    
    def solution(x):
        return (torch.sin(np.pi*x[:,0])*torch.sinh(np.pi*x[:,1])*x[:,2]+\
            torch.cos(np.pi*x[:,0])*torch.cos(np.pi*x[:,1])*torch.cos(np.pi*x[:,2])).reshape(-1,1)  

    # determine inverse
    def inverse(x):
        return torch.exp(-30*((x[:,0]-0.5)**2+(x[:,1]-0.5)**2+(x[:,2]-0.5)**2)).reshape(-1,1)          # 3D Case 2

    pde = PDE(equation, solution, inverse, args.xmin, args.xmax)

    # determine neural networks
    model_u = EITNet(args.num_channels, args.num_blocks, args.acti, dim=args.dim).to(args.device)
    model_s = EITNet(args.num_channels, args.num_blocks, args.acti, dim=args.dim).to(args.device)

    # determine loss function
    loss = Loss(pde, model_u, model_s)
    
    # loss function with g-fit term + data-fit term
    def loss_func(x, b, o, noisy_data, args):
        loss_i = loss.interior_loss(x)
        loss_g = loss.g_loss(x, g_func)
        loss_o = loss.observe_loss(o, noisy_data)
        loss_b = loss.dirichlet_boundary_loss(b,'s')
        loss_total = loss_i + \
                    args.lambda11*loss_g[0] + args.lambda12*loss_g[1] + args.lambda13*loss_g[2] + \
                    args.lambda2*loss_o + \
                    args.lambda3*loss_b
        loss_terms = (loss_i.detach(), loss_g[0].detach(), loss_g[1].detach(), loss_g[2].detach(), \
            loss_o.detach(), loss_b.detach())
        return loss_total, loss_terms

    # determine assessment metric
    def metric_func(pred, exact):
        metric = Metric(pred, exact)
        error = {}
        error['relative_error'] = metric.relative_l2_error().cpu()
        # error['l2_error'] = metric.l2_error()
        return error

    # build solver
    solver = Solver(model_u, model_s, loss_func, metric_func, pde, args)
    
    # training/testing/plotting
    if args.mode == 'train':
        # create folder if not exist
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            print('Create path : {}'.format(args.save_path))
        # create log and print to console
        LoggingPrinter(os.path.join(args.save_path, args.log_name+'.txt'))
        print('Current time is', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print_args(args)
        # start training
        solver.train()
        # write training arguments to csv file
        write_arg(args)
        print('Current time is', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    elif args.mode == 'test':
        solver.test()
    else:
        plot_result(args)
    

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(prog='DeepEIT', usage=print_usage())
    subparsers = parser.add_subparsers(dest = 'mode', required=True, help='train | test | plot')

    # shared paramters
    parser.add_argument('--device', type=str, default='cpu', help='cpu | cuda')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--num_channels', type=int, default=10, help='hidden layer width of network')
    parser.add_argument('--num_blocks', type=int, default=3, help='number of residual blocks of network')
    parser.add_argument('--acti', type=str, default='swish', help='activation function of the network')
    parser.add_argument('--dim', type=int, default=3, help='dimension of space')
    parser.add_argument('--xmin', type=float, default=0, help='lower bound of Omega')
    parser.add_argument('--xmax', type=float, default=1, help='upper bound of Omega')
    parser.add_argument('--csv_path', type=str, default='./result/params.csv', help='path of csv path to store training parameteres')
    parser.add_argument('--save_path', type=str, default='./result/test', help='saved path of the results')
    # training parameters
    subparser_train = subparsers.add_parser('train', help='training mode')
    subparser_train.add_argument('--num_interior_points', type=int, default=1000, help='number of interior points inside the domain Omega')
    subparser_train.add_argument('--num_boundary_points', type=int, default=100, help='number of points on the boundary partial Omega')
    subparser_train.add_argument('--num_observed_points', type=int, default=1000, help='number of observed points inside the domain Omega_0')
    subparser_train.add_argument('--num_validation_points', type=int, default=100, help='number of validation points')
    subparser_train.add_argument('--std', type=float, default=0, help='standard deviation of noise added to the observed data')
    subparser_train.add_argument('--omin', type=float, default=0.375, help='lower bound of Omega_0')
    subparser_train.add_argument('--omax', type=float, default=0.625, help='upper bound of Omega_0')
    subparser_train.add_argument('--lambda11', type=float, default=1e-2, help='hyper-parameter for the g loss')
    subparser_train.add_argument('--lambda12', type=float, default=1e-3, help='hyper-parameter for the g loss')
    subparser_train.add_argument('--lambda13', type=float, default=1e-3, help='hyper-parameter for the g loss')
    subparser_train.add_argument('--lambda2', type=float, default=20, help='hyper-parameter for the supervised loss')
    subparser_train.add_argument('--lambda3', type=float, default=100, help='hyper-parameter for the boundary loss')
    subparser_train.add_argument('--model_name', type=str, default='model', help='name of the models')
    subparser_train.add_argument('--lr_u', type=float, default=1e-2, help='learning rate of model_u')
    subparser_train.add_argument('--lr_s', type=float, default=1e-2, help='learning rate of model_s')
    subparser_train.add_argument('--decay_iters', type=int, default=2000, help='number of iterations to decay learning rate')
    subparser_train.add_argument('--save_iters', type=int, default=2000, help='number of iterations to save models')
    subparser_train.add_argument('--gamma', type=float, default=0.5, help='decay value of the learning rate')
    subparser_train.add_argument('--num_epochs', type=int, default=20000, help='number of epochs')
    subparser_train.add_argument('--print_iters', type=int, default=100, help='number of iterations to print statistics')
    subparser_train.add_argument('--loss_name', type=str, default='training_loss', help='name of the training loss')
    subparser_train.add_argument('--metric_name', type=str, default='validation_metric', help='name of the validation error')
    subparser_train.add_argument('--arg_name', type=str, default='training_arg', help='name of the arguments')
    subparser_train.add_argument('--if_save_checkpoint', action='store_false', help='save checkpoint')
    subparser_train.add_argument('--checkpoint_name', type=str, default='checkpoint', help='name of the checkpoint')
    subparser_train.add_argument('--log_name', type=str, default='log', help='name of the log file')
    subparser_train.add_argument('--lr_name', type=str, default='training_lr', help='name of the learning rate file')
    subparser_train.add_argument('--scheduler', type=str, default='step', help='type of the scheduler')
    # testing parameters
    subparser_test = subparsers.add_parser('test', help='testing mode')
    subparser_test.add_argument('--num_testing_points', type=int, default=200, help='number of testing points')
    subparser_test.add_argument('--model_name', type=str, default='model', help='name of the models')
    subparser_test.add_argument('--checkpoint_name', type=str, default='checkpoint', help='name of the checkpoint')
    subparser_test.add_argument('--error_name', type=str, default='testing_error', help='name of the error file')
    subparser_test.add_argument('--pred_name', type=str, default='testing_pred', help='name of testing predictions')
    subparser_test.add_argument('--not_save_result', action='store_false', help='not to save the results')
    # plotting parameters
    subparser_plot = subparsers.add_parser('plot', help='plotting mode')
    subparser_plot.add_argument('--num_plotting_points', type=int, default=200, help='number of plotting points')
    subparser_plot.add_argument('--plotting_point', type=int, default=100, help='coordinate of the point to be plotted')
    subparser_plot.add_argument('--pred_name', type=str, default='testing_pred', help='name of testing predictions to be plotted')
    subparser_plot.add_argument('--loss_name', type=str, default='training_loss', help='name of the training loss')
    subparser_plot.add_argument('--lr_name', type=str, default='training_lr', help='name of the learning rate')
    subparser_plot.add_argument('--metric_name', type=str, default='validation_metric', help='name of the validation error file')
    subparser_plot.add_argument('--plot_mode', type=str, default='imshow', help='countourf | imshow')
    subparser_plot.add_argument('--not_plot_loss', action='store_false', help='not to plot training loss')
    subparser_plot.add_argument('--not_plot_metric', action='store_false', help='not to plot validation metric')
    subparser_plot.add_argument('--not_plot_lr', action='store_false', help='not to plot learning rate')
    subparser_plot.add_argument('--not_plot_u', action='store_false', help='not to plot the predictions of model_u')
    subparser_plot.add_argument('--not_plot_s', action='store_false', help='not to plot the predictions of model_s')
    subparser_plot.add_argument('--not_save_plot', action='store_false', help='not to save the plot')

    args = parser.parse_args()

    # run the main function
    main(args)

