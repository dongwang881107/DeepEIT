import matplotlib
import torch
import os
import csv
import time
import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# print in both console and file
class LoggingPrinter:
    def __init__(self, filename):
        self.out_file = open(filename, "a")
        self.old_stdout = sys.stdout
        #this object will take over `stdout`'s job
        sys.stdout = self
    # executed when the user does a `print`
    def write(self, text): 
        self.old_stdout.write(text)
        self.out_file.write(text)
    # executed when `with` block begins
    def __enter__(self): 
        return self
    # executed when `with` block ends
    def __exit__(self, type, value, traceback): 
    # restore the original stdout object
        sys.stdout = self.old_stdout
    def flush(self):
        pass

# print usage 
def print_usage():
    return '''
    python main.py {train, test} [optional arguments]
    '''

# print arguments
def print_args(args):
    print('{:-^115s}'.format('Training parameters!'))
    count = 0
    for arg, value in args.__dict__.items():
        count = count + 1
        if count == 2:
            print('{:>25} = {:<25}'.format(arg, value))
            count = 0
        else:
            print('{:>25} = {:<25}'.format(arg, value), end='| ')
    if count == 1:
        print('\n')

# print statistics
def print_stat(epoch, loss, metric_u, metric_s, start_time):
    print('epoch = {:<6} | loss = {:<10.4f}'.format(epoch+1, loss[epoch]), end="| ")
    keys = list(metric_u[0].keys())
    for key in keys:
        txt = key+'_u = {:<8.4f} | '+key+'_sigma = {:<8.4f}'
        print(txt.format(metric_u[-1][key], metric_s[-1][key]), end="| ")
    print('time = {:<8.2f}'.format(time.time()-start_time))

# print errors
def print_error(error_u, error_s):
    keys = list(error_u.keys())
    for key in keys:
        print('{:<20} = {:.4f}'.format(key+'_u', error_u[key]))
        print('{:<20} = {:.4f}'.format(key+'_sigma', error_s[key]))

# save model parameters
def save_model(model_u, model_s, args):
    model_u_path = os.path.join(args.save_path, args.model_name+'_u.pkl')
    model_s_path = os.path.join(args.save_path, args.model_name+'_sigma.pkl')
    torch.save(model_u.state_dict(), model_u_path)
    torch.save(model_s.state_dict(), model_s_path)
    print('{:>40} => {:<40}'.format('Model u saved in', model_u_path))
    print('{:>40} => {:<40}'.format('Model sigma saved in', model_s_path))

# save checkpoint
def save_checkpoint(model_u, model_s, optimizer, epoch, args):
    if args.if_save_checkpoint:
        checkpoint_path = os.path.join(args.save_path, 'checkpoint')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            print('Create checkpoint path : {}'.format(checkpoint_path))
        checkpoint = {'model_u':model_u.state_dict(), 'model_s':model_s.state_dict(),\
            'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(checkpoint, checkpoint_path+'/checkpoint_{}.pkl'.format(epoch+1))
        print('{:>40} => {:<40}'.format('Checkpoint saved in', checkpoint_path+'/checkpoint_{}.pkl'.format(epoch+1)))

# load checkpoint
def load_checkpoint(model_u, model_s, optimizer, args):
    checkpoint_path = os.path.join(args.save_path, 'checkpoint', args.checkpoint_name+'.pkl')
    if os.path.exists(checkpoint_path):
        if args.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model_u.load_state_dict(checkpoint['model_u'])
        model_s.load_state_dict(checkpoint['model_s'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('{} => {}'.format('Successfully loaded checkpoint', start_epoch))
    else:
        start_epoch = 0
        print('No checkpoint found! Training from epoch 0!')
    return start_epoch

# save training loss
def save_loss(loss_total, loss_terms, args):
    loss_path = os.path.join(args.save_path, args.loss_name+'.npy')
    losses = np.zeros([len(loss_terms), len(loss_terms[0])+1])
    for i in range(len(loss_terms)):
        for j in range(len(loss_terms[0])+1):
            if j == 0:
                losses[i][j] = loss_total[i]
            else:
                losses[i][j] = loss_terms[i][j-1]
    np.save(loss_path, losses)
    print('{:>40} => {:<40}'.format('Training loss saved in', loss_path))


# save training loss
def save_lr(lr, args):
    lr_path = os.path.join(args.save_path, args.lr_name+'.npy')
    np.save(lr_path, lr)
    print('{:>40} => {:<40}'.format('Learning rate saved in', lr_path))

# save validation metric
def save_metric(metric_u, metric_s, args):
    metric_u_path = os.path.join(args.save_path, args.metric_name+'_u.npy')
    metric_s_path = os.path.join(args.save_path, args.metric_name+'_sigma.npy')
    np.save(metric_u_path, metric_u)
    np.save(metric_s_path, metric_s)
    print('{:>40} => {:<40}'.format('Validation metric of u saved in', metric_u_path))
    print('{:>40} => {:<40}'.format('Validation metric of sigma saved in', metric_s_path))

# load statistics
def load_statistics(start_epoch, args):
    if start_epoch == 0:
        loss_total = []
        loss_terms = []
        metric_u = []
        metric_s = []
        lr = []
    else:
        loss_path = os.path.join(args.save_path, args.loss_name+'.npy')
        losses = np.load(loss_path)
        loss_total = list(losses[:,0])
        loss_terms = []
        for i in range(len(losses)):
            loss_terms.append(tuple(losses[i,1:losses.shape[1]]))

        metric_u_path = os.path.join(args.save_path, args.metric_name+'_u.npy')
        metric_s_path = os.path.join(args.save_path, args.metric_name+'_sigma.npy')
        metric_u = list(np.load(metric_u_path, allow_pickle='TRUE'))
        metric_s = list(np.load(metric_s_path, allow_pickle='TRUE'))

        lr_path = os.path.join(args.save_path, args.lr_name+'.npy')
        lr = list(np.load(lr_path))
    return loss_total, loss_terms, metric_u, metric_s, lr

# save training arguments 
def save_arg(args):
    arg_path = os.path.join(args.save_path, args.arg_name+'.txt')
    argsDict = args.__dict__
    with open(arg_path,'w') as file:
        for arg, value in argsDict.items():
            file.writelines(arg+':'+str(value)+'\n')
    print('{:>40} => {:<40}'.format('Training arguments saved in', arg_path))

# write training arguments to csv file
def write_arg(args):
    argsDict = args.__dict__
    title = []
    params = []
    useless_columns = ['mode', 'device', 'seed', 'xmin', 'xmax', 'csv_path', 'model_name', 'loss_name',\
         'save_iters', 'print_iters', 'metric_name', 'arg_name', 'if_save_checkpoint', 'log_name', 'lr_name']
    for arg, value in argsDict.items():
        if arg not in useless_columns:
            title.append(arg)
            params.append(str(value))
    # add new lines to csv file
    if not os.path.exists(args.csv_path):
        with open(args.csv_path, 'a') as f:
            write = csv.writer(f)
            write.writerow(title)
            write.writerow(params)
    else:
        with open(args.csv_path, 'a') as f:
            write = csv.writer(f)
            write.writerow(params)
    # remove duplicate rows and useless columns
    frame = pd.read_csv(args.csv_path, engine='python')
    frame.drop_duplicates(inplace=True)
    frame.to_csv(args.csv_path, index=False)

# load model parameters
def load_model(model_u, model_s, args):
    print('{:-^115s}'.format('Loading pre-trained models!'))
    checkpoint_path = os.path.join(args.save_path, 'checkpoint', args.checkpoint_name+'.pkl')
    if os.path.exists(checkpoint_path):
        if args.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model_u.load_state_dict(checkpoint['model_u'])
        model_s.load_state_dict(checkpoint['model_s'])
        print('Loading checkpoint ...')
    else:
        model_u_path = os.path.join(args.save_path, args.model_name+'_u.pkl')
        model_s_path = os.path.join(args.save_path, args.model_name+'_sigma.pkl')
        if args.device == 'cuda':
            model_u.load_state_dict(torch.load(model_u_path))
            model_s.load_state_dict(torch.load(model_s_path))
        else:
            model_u.load_state_dict(torch.load(model_u_path, map_location=torch.device('cpu')))
            model_s.load_state_dict(torch.load(model_s_path, map_location=torch.device('cpu')))
        print('Loading model u ...')
        print('Loading model sigma ...')

# save testing prediction
def save_pred(pred_u, exact_u, pred_s, exact_s, args):
    testing_result_path = os.path.join(args.save_path, args.pred_name+'.npz')
    np.savez(testing_result_path,
            pred_u=pred_u, exact_u=exact_u, 
            pred_sigma=pred_s, exact_sigma=exact_s)
    print('{:>40} => {:<40}'.format('Testing predictions saved in', testing_result_path))

# save testing error
def save_error(error_u, error_s, args):
    error_path = os.path.join(args.save_path, args.error_name+'.txt')
    with open(error_path, 'w') as file:
        for key, value in error_u.items():
            file.writelines(key+' of u:'+str(value)+'\n')
        for key, value in error_s.items():
            file.writelines(key+' of sigma:'+str(value)+'\n')
    print('{:>40} => {:<40}'.format('Testing errors saved in', error_path))

# plot training loss
def plot_loss(args):
    # load training loss
    loss_path = os.path.join(args.save_path, args.loss_name+'.npy')
    losses = np.load(loss_path)

    # plot total training loss
    fs = 18
    fig = plt.figure()
    plt.xlabel('Epoch', fontsize=fs)
    plt.ylabel('Training Loss', fontsize=fs)
    plt.plot(losses[:,0], linewidth=1.0)
    plt.yscale('log')
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')
    if args.not_save_plot:
        plot_path = os.path.join(args.save_path, args.loss_name+'_total.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print('{:>40} => {:<40}'.format('Total training loss plot saved in', plot_path))

    # plot losses of terms
    fs = 18
    fig = plt.figure()
    plt.xlabel('Epoch', fontsize=fs)
    plt.ylabel('Term Losses', fontsize=fs)
    for j in range(1, losses.shape[1]):
        plt.plot(losses[:,j], linewidth=1.0)
        plt.yscale('log') 
    plt.legend(['PDE-fit','g_lap-fit', 'g_grad-fit', 'g_ori-fit', 'data-fit', '$\sigma-\mathrm{fit}$'])
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')
    if args.not_save_plot:
        plot_path = os.path.join(args.save_path, args.loss_name+'_term.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print('{:>40} => {:<40}'.format('Training loss terms plot saved in', plot_path))

# plot validation metrics
def plot_metric(args):
    # load validation metrics
    metric_u_path = os.path.join(args.save_path, args.metric_name+'_u.npy')
    metric_s_path = os.path.join(args.save_path, args.metric_name+'_sigma.npy')
    metric_u = np.load(metric_u_path, allow_pickle='TRUE')
    metric_s = np.load(metric_s_path, allow_pickle='TRUE')
    # extract data from dictionary
    keys = list(metric_u[0].keys())
    metrics = np.zeros([len(metric_u), len(keys)*2])
    for i in range(len(metric_u)):
        for j in range(len(keys)):
            metrics[i,2*j] = metric_u[i][keys[j]]
            metrics[i,2*j+1] = metric_s[i][keys[j]]
    # plot validation metrics
    for j in range(len(keys)):
        fs = 18
        fig = plt.figure()
        plt.xlabel('Epoch', fontsize=fs)
        plt.ylabel(keys[j].replace('_',' ').title(), fontsize=fs)
        plt.plot(metrics[:,2*j], linewidth=1.0)
        plt.plot(metrics[:,2*j+1], linewidth=1.0)
        plt.legend(['u','$\sigma$'])
        plt.yscale('log')
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')
        if args.not_save_plot:
            plot_path = os.path.join(args.save_path, keys[j]+'.png')
            fig.savefig(plot_path, bbox_inches='tight')
            print('{:>40} => {:<40}'.format(keys[j].capitalize()+' plot saved in', plot_path))

# plot testing predictions
def plot_pred(model_name, args, dim=0):
    # load testing results
    result_path = os.path.join(args.save_path, args.pred_name+'.npz')
    result = np.load(result_path)
    pred  = result['pred_'+model_name].reshape([args.num_plotting_points]*args.dim)
    exact = result['exact_'+model_name].reshape([args.num_plotting_points]*args.dim)
    if args.dim == 3:
        assert(args.plotting_point>=0 & args.plotting_point<args.num_plotting_points)
        if dim == 0:
            exact = exact[args.plotting_point,:,:]
            pred = pred[args.plotting_point,:,:]
        elif dim == 1:
            exact = exact[:,args.plotting_point,:] 
            pred = pred[:,args.plotting_point,:]
        elif dim == 2:
            exact = exact[:,:,args.plotting_point]
            pred = pred[:,:,args.plotting_point]
    
    # plot testing results
    fs = 18
    cmap = 'hot'
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
    vmin = np.min([np.min(pred), np.min(exact)])
    vmax = np.max([np.max(pred), np.max(exact)])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if (args.plot_mode == 'contourf') & (args.dim == 2):
        # meshgrid
        x1 = torch.linspace(0., 1., args.num_plotting_points)
        x2 = torch.linspace(0., 1., args.num_plotting_points)
        [X1,X2] = torch.meshgrid(x1,x2)
        im1 = ax1.contourf(X1, X2, exact, cmap=cmap, norm=norm)
        im2 = ax2.contourf(X1, X2, pred, cmap=cmap, norm=norm)
        im3 = ax3.contourf(X1, X2, np.abs(exact-pred), cmap=cmap)
    elif args.plot_mode == 'imshow':
        im1 = ax1.imshow(exact, cmap=cmap, norm=norm)
        im2 = ax2.imshow(pred, cmap=cmap, norm=norm)
        im3 = ax3.imshow(np.abs(exact-pred), cmap=cmap)
    else:
        print('contourf (2D only) | imshow (2D & 3D)')
        sys.exit(0)

    if model_name =='sigma':
        ax1.set_title('$\sigma_\mathrm{exact}$', fontsize=fs)
        ax2.set_title('$\sigma_\mathrm{proposed}$', fontsize=fs)
        ax3.set_title('|$\sigma_\mathrm{proposed}$-$\sigma_\mathrm{exact}$|', fontsize=fs)
    else:
        ax1.set_title('$'+model_name+'_\mathrm{exact}$', fontsize=fs)
        ax2.set_title('$'+model_name+'_\mathrm{proposed}$', fontsize=fs)
        ax3.set_title('|'+'$'+model_name+'_\mathrm{proposed}$-$'+model_name+'_\mathrm{exact}$|', fontsize=fs)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    
    fraction = 0.045
    fig.colorbar(im1, ax=ax1, fraction=fraction)
    fig.colorbar(im1, ax=ax2, fraction=fraction)
    fig.colorbar(im3, ax=ax3, fraction=fraction)

    # plt.ion()
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')
    if args.not_save_plot:
        if args.dim == 2:
            plot_path = os.path.join(args.save_path, args.pred_name+'_'+model_name+'.png')
        elif args.dim == 3:
            plot_path = os.path.join(args.save_path, args.pred_name+'_'+model_name+'_'+str(dim)+'.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print('{:>40} => {:<40}'.format('Prediction of model '+model_name+' saved in', plot_path))

# plot learning rate
def plot_lr(args):
    # load learning rate
    lr_path = os.path.join(args.save_path, args.lr_name+'.npy')
    lr = np.load(lr_path)

    # plot total training loss
    fs = 18
    fig = plt.figure()
    plt.xlabel('Epoch', fontsize=fs)
    plt.ylabel('Learning Rate', fontsize=fs)
    plt.plot(lr, linewidth=1.0)
    plt.yscale('log')
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')
    if args.not_save_plot:
        plot_path = os.path.join(args.save_path, args.lr_name+'.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print('{:>40} => {:<40}'.format('Learning rate plot saved in', plot_path))

# plot all the results
def plot_result(args):
    print('{:-^115s}'.format('Plotting!'))
    if args.not_plot_loss == True:
        plot_loss(args)
    if args.not_plot_metric == True:
        plot_metric(args)
    if args.not_plot_lr == True:
        plot_lr(args)
    if args.not_plot_u == True:
        plot_pred('u', args)
        if args.dim == 3:
            plot_pred('u',args,1)
            plot_pred('u',args,2)
    if args.not_plot_s == True:
        plot_pred('sigma', args)
        if args.dim == 3:
            plot_pred('sigma',args,1)
            plot_pred('sigma',args,2)
    print('{:-^115s}'.format('Done!'))
