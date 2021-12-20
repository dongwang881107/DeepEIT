import matplotlib
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# save model parameters
def save_model(model_u, model_f, args):
    model_u_path = os.path.join(args.save_path, args.model_name+'_u.pkl')
    model_f_path = os.path.join(args.save_path, args.model_name+'_f.pkl')
    torch.save(model_u.state_dict(), model_u_path)
    torch.save(model_f.state_dict(), model_f_path)
    print('Model u saved in {}'.format(model_u_path))
    print('Model f saved in {}'.format(model_f_path))

# save training loss
def save_loss(loss, args):
    loss_path = os.path.join(args.save_path, args.loss_name+'.npy')
    np.save(loss_path, loss)
    print('Training loss saved in {}'.format(loss_path))

# save training arguments
def save_arg(args):
    arg_path = os.path.join(args.save_path, args.arg_name+'.txt')
    argsDict = args.__dict__
    with open(arg_path,'w') as file:
        for arg, value in argsDict.items():
            file.writelines(arg+':'+str(value)+'\n')
    print('Training arguments saved in {}'.format(arg_path))

# load model parameters
def load_model(model_u, model_f, args):
    model_u_path = os.path.join(args.save_path, args.model_name+'_u.pkl')
    model_f_path = os.path.join(args.save_path, args.model_name+'_f.pkl')
    model_u.load_state_dict(torch.load(model_u_path))
    model_f.load_state_dict(torch.load(model_f_path))
    print('Load model u')
    print('Load model f')

# save testing results
def save_result(u_relative_error, f_relative_error, u_l2_error, f_l2_error, u_pred, u_exact, f_pred, f_exact, args):
    testing_result_path = os.path.join(args.save_path, args.testing_result_name+'.npz')
    np.savez(testing_result_path, 
            u_relative_error=u_relative_error, f_relative_error=f_relative_error,
            u_l2_error=u_l2_error, f_l2_error=f_l2_error,
            u_pred=u_pred, u_exact=u_exact, 
            f_pred=f_pred, f_exact=f_exact)
    print('Testing resuts saved in {}'.format(testing_result_path))

# plot training loss
def plot_loss_history(args):
    # load training loss
    loss_path = os.path.join(args.save_path, args.loss_name+'.npy')
    loss = np.load(loss_path)
    # plot training loss
    fs = 18
    fig = plt.figure()
    plt.xlabel('Epoch', fontsize=fs)
    plt.ylabel('Training Loss', fontsize=fs)
    plt.plot(loss)

    plt.show()
    if args.not_save_plot:
        plot_path = os.path.join(args.save_path, args.loss_name+'.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print('Loss plot saved in {}'.format(plot_path))

# plot predictions
def plot_prediction(args, model_name):
    # load testing results
    result_path = os.path.join(args.save_path, args.plotting_result_name+'.npz')
    result = np.load(result_path)
    pred = result[model_name+'_pred'].reshape(args.num_testing_points,args.num_testing_points)
    exact = result[model_name+'_exact'].reshape(args.num_testing_points,args.num_testing_points)
    # meshgrid
    x1 = torch.linspace(0., 1., args.num_testing_points)
    x2 = torch.linspace(0., 1., args.num_testing_points)
    [X1,X2] = torch.meshgrid(x1, x2)
    # plot testing results
    fs = 18
    vmin = np.min([np.min(pred), np.min(exact)])
    vmax = np.max([np.max(pred), np.max(exact)])
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if args.plot_mode == 'contourf':
        im1 = ax1.contourf(X1, X2, exact, cmap='jet', norm=norm)
        im2 = ax2.contourf(X1, X2, pred, cmap='jet', norm=norm)
        im3 = ax3.contourf(X1, X2, np.abs(exact-pred), cmap='jet', norm=norm)
    elif args.plot_mode == 'imshow':
        im1 = ax1.imshow(exact, cmap='jet', norm=norm)
        im2 = ax2.imshow(pred, cmap='jet', norm=norm)
        im3 = ax3.imshow(np.abs(exact-pred), cmap='jet', norm=norm)
    else:
        print('contourf | imshow')
        sys.exit(0)

    ax1.set_title(model_name+'_exact', fontsize=fs)
    ax2.set_title(model_name+'_pred', fontsize=fs)
    ax3.set_title('|'+model_name+'_pred-'+model_name+'_exact|', fontsize=fs)
    fig.subplots_adjust(right=0.9)
    position = fig.add_axes([0.92, 0.12, 0.015, 0.78])
    fig.colorbar(im1, cax=position)

    plt.show()
    if args.not_save_plot:
        plot_path = os.path.join(args.save_path, args.plot_name+'_'+model_name+'.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print('Prediction of model {} plot saved in {}'.format(model_name, plot_path))

def plot_result(args):
    if args.not_plot_loss == True:
        plot_loss_history(args)
    if args.not_plot_u == True:
        plot_prediction(args, 'u')
    if args.not_plot_f == True:
        plot_prediction(args, 'f')

