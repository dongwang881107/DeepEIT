import os
import argparse
import warnings

from solver import *
from data import *
from postprocessing import *

def main(args):
    # set seed and data type
    torch.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.DoubleTensor')
    warnings.filterwarnings('ignore')

    # create folder if not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    # training/testing/plotting
    if args.mode == 'train':
        # generate supervised data
        supervised_dataset = SupervisedPoints(args.num_supervised_points, args.lower, args.upper)
        # build solver
        solver = Solver(supervised_dataset, args)
        # training
        solver.train()
    elif args.mode == 'test':
        # generate testing data
        testing_dataset = SupervisedPoints(args.num_testing_points, 0., 1., mode='test')
        # bulid solver
        solver = Solver(testing_dataset, args)
        # testing
        solver.test()
    else:
        plot_loss_history()

# usage function
def usage():
    return '''
    python main.py {train, test} [optional arguments]
    '''

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(prog='DL-EIT', usage=usage())
    subparsers = parser.add_subparsers(dest = 'mode', required=True, help='train | test | plot')

    # shared paramters
    parser.add_argument('--device', type=str, default='cpu', help='cpu | cuda')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--num_channels', type=int, default=6, help='hidden layer width of network')
    parser.add_argument('--num_blocks', type=int, default=3, help='number of residual blocks of network')
    parser.add_argument('--acti', type=str, default='swish', help='activation function of the network')
    parser.add_argument('--save_path', type=str, default='./result', help='saved path of the results')
    parser.add_argument('--model_name', type=str, default='model', help='name of the models')
    # training parameters
    subparser_train = subparsers.add_parser('train', help='training mode')
    subparser_train.add_argument('--num_interior_points', type=int, default=1000, help='number of interior points inside the domain Omega')
    subparser_train.add_argument('--num_boundary_points', type=int, default=100, help='number of points on the boundary partial Omega')
    subparser_train.add_argument('--num_supervised_points', type=int, default=1000, help='number of supervised points inside the domain Omega_0')
    subparser_train.add_argument('--lower', type=float, default=0.5, help='lower bound of Omega_0')
    subparser_train.add_argument('--upper', type=float, default=0.75, help='upper bound of Omega_0')
    subparser_train.add_argument('--lambda1', type=float, default=1, help='hyper-parameter for the boundary loss')
    subparser_train.add_argument('--lambda2', type=float, default=100, help='hyper parameter for the supervised loss')
    subparser_train.add_argument('--lr_u', type=float, default=1e-3, help='learning rate of model_u')
    subparser_train.add_argument('--lr_f', type=float, default=1e-3, help='learning rate of model_f')
    subparser_train.add_argument('--decay_iters', type=int, default=1000, help='number of iterations to decay learning rate')
    subparser_train.add_argument('--gamma', type=float, default=0.5, help='decay value of the learning rate')
    subparser_train.add_argument('--num_epochs', type=int, default=20000, help='number of epochs')
    subparser_train.add_argument('--print_iters', type=int, default=100, help='number of iterations to print statistics')
    subparser_train.add_argument('--loss_name', type=str, default='training_loss', help='name of the training loss')
    subparser_train.add_argument('--arg_name', type=str, default='training_arg', help='name of the arguments')
    # testing parameters
    subparser_test = subparsers.add_parser('test', help='testing mode')
    subparser_test.add_argument('--num_testing_points', type=int, default=500, help='number of testing points')
    subparser_test.add_argument('--testing_result_name', type=str, default='testing_result', help='name of testing results')
    # plotting parameters
    subparser_plot = subparsers.add_parser('plot', help='plotting mode')
    subparser_plot.add_argument('--plotting_result_name', type=str, default='testing_result', help='name of testing results')

    args = parser.parse_args()

    # run the main function
    main(args)
