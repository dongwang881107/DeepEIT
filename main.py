import os
import sys
import argparse
import torch.optim as optim

from solver import *
from data import *
from model import *
from loss import *

def main(args):
    # set seed and data type
    torch.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.DoubleTensor')

    # create folder if not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    # training/testing
    if args.mode == 'train':
        # generate supervised data
        supervised_dataset = SupervisedPoints(args.num_supervised_points, args.lower, args.upper)
        # generate networks
        model_u = ResNet(args.num_channels, args.num_blocks).to(args.device)
        model_f = ResNet(args.num_channels, args.num_blocks).to(args.device)
        # loss function and optimizer
        criterion = compute_loss
        optimizer = optim.Adam([{'params':model_u.parameters(),'lr':args.lr}, {'params':model_f.parameters(),'lr':args.lr}])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_iters, gamma=args.gamma)
        # build solver
        args.criterion = criterion
        args.optimizer = optimizer
        args.scheduler = scheduler
        solver = Solver(supervised_dataset, model_u, model_f, args)
        # training
        solver.train()
    else:
        # generate testing data
        testing_dataset = SupervisedPoints(args.num_testing_points, 0, 1)
        # load networks
        model_u = ResNet(args.num_channels, args.num_blocks).to(args.device)
        model_f = ResNet(args.num_channels, args.num_blocks).to(args.device)
        model_u_path = os.path.join(args.save_path, args.model_name+'_u.pkl')
        model_f_path = os.path.join(args.save_path, args.model_name+'_f.pkl')
        model_u.load_state_dict(torch.load(model_u_path))
        model_f.load_state_dict(torch.load(model_f_path))
        # bulid solver
        solver = Solver(testing_dataset, model_u, model_f, args)
        # testing
        solver.test()

# usage function
def usage():
    return '''
    python main.py {train, test} [optional arguments]
    '''

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(prog='DL-EIT', usage=usage())
    subparsers = parser.add_subparsers(dest = 'mode', required=True, help='train | test')

    # shared paramters
    parser.add_argument('--device', type=str, default='cpu', help='cpu | cuda')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--num_channels', type=int, default=6, help='hidden layer width of network')
    parser.add_argument('--num_blocks', type=int, default=3, help='number of residual blocks of network')
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
    subparser_train.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    subparser_train.add_argument('--decay_iters', type=int, default=1000, help='number of iterations to decay learning rate')
    subparser_train.add_argument('--gamma', type=float, default=0.5, help='decay value of the learning rate')
    subparser_train.add_argument('--num_epochs', type=int, default=20000, help='number of epochs')
    subparser_train.add_argument('--print_iters', type=int, default=100, help='number of iterations to print statistics')
    # testing parameters
    subparser_test = subparsers.add_parser('test', help='testing mode')
    subparser_test.add_argument('--num_testing_points', type=int, default=1000, help='number of testing points')

    args = parser.parse_args()

    # run the main function
    main(args)
