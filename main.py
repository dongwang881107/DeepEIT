import os
import argparse
import torch.optim as optim

from solver import *
from data import *
from model import *
from loss import *

def main(args):
    # create folder if not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    # generate supervised data
    supervised_dataset = SupervisedPoints(args.num_supervised_points, args.lower, args.upper)

    # generate networks
    model_u = ResNet(args.num_channels)
    model_f = ResNet(args.num_channels)

    # loss function and optimizer
    criterion = compute_loss
    optimizer = optim.Adam([{'params':model_u.parameters(),'lr':args.lr}, {'params':model_f.parameters(),'lr':args.lr}])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_iters, gamma=args.gamma)

    # build solver
    solver = Solver(supervised_dataset, model_u, model_f, criterion, optimizer, scheduler, args)

    # training/testing
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    else:
        print('train | test')

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train | test')
    parser.add_argument('--num_interior_points', type=int, default=2000, help='number of interior points inside the domain Omega')
    parser.add_argument('--num_boundary_points', type=int, default=100, help='number of points on the boundary partial Omega')
    parser.add_argument('--num_supervised_points', type=int, default=1000, help='number of supervised points inside the domain Omega_0')
    parser.add_argument('--lower', type=float, default=0.5, help='lower bound of Omega_0')
    parser.add_argument('--upper', type=float, default=0.75, help='upper bound of Omega_0')
    parser.add_argument('--num_channels', type=int, default=5, help='hidden layer width of network')
    parser.add_argument('--lambda1', type=float, default=1, help='')
    parser.add_argument('--lambda2', type=float, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay_iters', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_epochs', type=int, default=20000)
    parser.add_argument('--print_iters', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='./result')
    parser.add_argument('--model_name', type=str, default='model')

    args = parser.parse_args()

    # run the main function
    main(args)
