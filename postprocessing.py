import torch
import os
import numpy as np

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
def save_result(u_relative_error, f_relative_error, u_l2_error, f_l2_error, 
    s, u_pred, u_exact, f_pred, f_exact, args):
    testing_result_path = os.path.join(args.save_path, args.testing_result_name+'.npz')
    np.savez(testing_result_path, 
            u_relative_error=u_relative_error, f_relative_error=f_relative_error,
            u_l2_error=u_l2_error, f_l2_error=f_l2_error,
            s=s, u_pred=u_pred, u_exact=u_exact, 
            f_pred=f_pred, f_exact=f_exact)
    print('Testing resuts saved in {}'.format(testing_result_path))

# plot training loss
def plot_loss_history():
    return True

# plot predictions
def plot_prediction():
    return True

