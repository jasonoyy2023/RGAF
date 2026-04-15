"""
Masked Autoregressive Flow for Density Estimation
arXiv:1705.07057v4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
import numpy as np
import pickle
import os
import math
import argparse
import pprint
import copy
from metrics import masked_mae, masked_mse, masked_mape_np, _mape
from torch.utils.data import DataLoader, TensorDataset
from GFLOWLib import GMADE,GMAF
import pytorch_forecasting 
from torch.optim import lr_scheduler

test_npz_path='RGAF_METRLAtest.npz'
thredhold=1000.0
parser = argparse.ArgumentParser()
# action
parser.add_argument('--train', default=True, help='Train a flow.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--data_dir', default='./data/', help='Location of datasets.')
parser.add_argument('--time_steps', type=int, default=12, help='Number of time steps to use.')
parser.add_argument('--pred_steps', type=int, default=12, help='Number of time steps to use.')
parser.add_argument('--output_dir', default='./results/')
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
# data
parser.add_argument('--dataset', default='metr-la', help='Which dataset to use.')
parser.add_argument('--flip_toy_var_order', action='store_true', help='Whether to flip the toy dataset variable order to (x2, x1).')
parser.add_argument('--seed', type=int, default=1, help='Random seed to use.')
# model
parser.add_argument('--model', default='gmaf', help='Which model to use: made, maf.')
# made parameters
parser.add_argument('--n_blocks', type=int, default=5, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=500, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=2, help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='tanh', help='What activation function to use in the MADEs.')
parser.add_argument('--input_order', type=str, default='sequential', help='What input order to use (sequential | random).')
# training params
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train for.')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=100, help='How often to show loss statistics and save samples.')


# --------------------
# Train and evaluate
# --------------------
#训练代码待论文发表后加入
    # 同步写入结果文件，记录 4 个变量
    
    outsets=os.path.join(args.output_dir,datasetname)
    with open(os.path.join(outsets, 'eval_results_'+str(args.pred_steps)+'.txt'), 'a') as f:
        f.write(log_str + '\n')

@torch.no_grad()
def evaluate(model, dataloader, args, adj):
    model.eval()
    rmse_12_list = []
    mae_12_list = []
    mape_12_list = []

    mape_12_list2 = []
    y_true_list=[]
    y_pred_list=[]

    seq_len=args.pred_steps
    for i, data in enumerate(dataloader):
        x, y = data
        x = x.to(args.device)
        y = y.to(args.device)
        y_hat = model.inverse(x, adj)
  
        b,t, n = y_hat.shape
        y_true=y[:,:,:,0]
        y_pred=y_hat

        y_true=y_true.reshape(b,seq_len*n)
        y_pred=y_pred.reshape(b,seq_len*n) 

    
        y_pred_list.append(y_pred.cpu().numpy())
        y_true_list.append(y_true.cpu().numpy())

        rmse_12=pytorch_forecasting.metrics.RMSE()(  y_pred, y_true)
        mae_12=pytorch_forecasting.metrics.MAE()(y_pred,y_true)
        #由于y_true中存在0值，导致MAPE计算时出现除以0的情况，因此在计算MAPE时需要进行特殊处理
        t2=torch.clamp(torch.abs(y_true), min=4.0)
        mape_12=pytorch_forecasting.metrics.MAPE()(y_pred,t2)
        if not torch.isnan(mape_12) and mape_12 < thredhold:
            mape_12_list.append(mape_12.item())

        rmse_12_list.append(rmse_12.item())
        mae_12_list.append(mae_12.item())
        # mape_12_list.append(mape_12.item())

    
    np.savez_compressed(test_npz_path, y_true=np.array(y_true_list),y_pred=np.array(y_pred_list))

    # 拼接所有 batch 的结果
    eval_mae_12=np.mean(mae_12_list)
    eval_rmse_12=np.mean(rmse_12_list)
    eval_mape_12=np.mean(mape_12_list)


    # eval_mape_12_2=np.mean(mape_12_list2)
 
    output = 'Evaluate (masked_MAE: {:.4f}, masked_MSE: {:.4f}, masked_MAPE: {:.4f}'.format(
         eval_mae_12, eval_rmse_12, eval_mape_12*100 )
    print(output)
    outsets=os.path.join(args.output_dir,datasetname)
    with open(os.path.join(outsets, 'eval_results_'+str(args.pred_steps)+'.txt'), 'a') as f:
        f.write(output + '\n')


def train_and_evaluate(model, train_loader, test_loader, optimizer, args, adj ):
    # 训练开始前，将超参数和 loss 公式写入日志文件
    config_header = (
        '\n' + '='*60 + '\n'
        '[Training Config]\n'
        '  n_blocks     : {}\n'
        '  n_hidden     : {}\n'
        '  hidden_size  : {}\n'
        '  n_epochs     : {}\n'
        '  batch_size   : {}\n'
        '  lr           : {}\n'
        '  log_interval : {}\n'
        '  activation   : {}\n'
        '  input_order  : {}\n'
        '[Loss Formula]\n'
        '  loss = -0.1 * logP + 10.0 * recon_y + 1.0 * recon_x\n'
        + '='*60 + '\n'
    ).format(
        args.n_blocks, args.n_hidden, args.hidden_size,
        args.n_epochs, args.batch_size, args.lr,
        args.log_interval, args.activation_fn, args.input_order
    )
    print(config_header)
    outsets=os.path.join(args.output_dir,datasetname)
    model_path=os.path.join(outsets, 'model_state'+str(args.pred_steps)+'.pt')
    if os.path.exists(model_path)==True:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    with open(os.path.join(outsets, 'eval_results_'+str(args.pred_steps)+'.txt'), 'w') as f:
        f.write(config_header)
    f.close()
 
    
    evaluate(model, test_loader,  args, adj)

    torch.save(model.state_dict(), os.path.join(outsets, 'model_state'+str(args.pred_steps)+'.pt'))


if __name__ == '__main__':
    args = parser.parse_args()
    datasetname=args.dataset
    if not os.path.isdir(args.output_dir+datasetname):
        os.makedirs(args.output_dir+datasetname)

    # setup device
    # args.device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.device=torch.device("mps")
    torch.manual_seed(args.seed)
    # if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # load data
    # pemsbay_train = np.load('pemsbay/train_640.npz')
    # pemsbay_test = np.load('pemsbay/test_640.npz')
    
    tfile=datasetname+'/train_'+str(args.pred_steps)+'.npz'
    sfile=datasetname+'/test_'+str(args.pred_steps)+'.npz'

    pemsbay_train = np.load(tfile)
    pemsbay_test = np.load(sfile)    
    sensor_nums = pemsbay_train['x'].shape[2]
    print(f"Sensor numbers: {sensor_nums}")

    # 加载邻接矩阵
    adj_data = pickle.load(open(datasetname+'/'+datasetname+'_adj.pkl', 'rb'), encoding='latin1')
    adj_mx = torch.from_numpy(adj_data[2].astype(np.float32))
    adj_mx = adj_mx.to(args.device)
    
    # 提取数据
    trainX = pemsbay_train['x']
    trainY = pemsbay_train['y'] 
    testX = pemsbay_test['x'] 
    testY = pemsbay_test['y'] 

    trainX = torch.from_numpy(trainX.astype(np.float32))
    trainY = torch.from_numpy(trainY.astype(np.float32))
    testX = torch.from_numpy(testX.astype(np.float32))
    testY = torch.from_numpy(testY.astype(np.float32))    
    
    # trainX = torch.squeeze(trainX)
    # testX = torch.squeeze(testX)
    # trainY = torch.squeeze(trainY)
    # testY = torch.squeeze(testY)

    train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(testX, testY), batch_size=args.batch_size, shuffle=False, drop_last=True)

    args.input_size = sensor_nums
    args.input_dims = sensor_nums

    # model
    if args.model == 'gmade':
        model = GMADE(args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                     args.activation_fn, args.input_order)
    elif args.model == 'gmaf':
        model = GMAF(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden,
                    args.activation_fn, args.input_order,batch_size=args.batch_size,timesteps=args.pred_steps, senums=sensor_nums)
    else:
        raise ValueError('Unrecognized model.')

    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    if args.train:
        train_and_evaluate(model, train_loader, test_loader, optimizer, args, adj_mx)
    