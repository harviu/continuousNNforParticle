import random
import os
import argparse
import math

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler,RandomSampler

from models.deepSDF import AD_SDF
from utils import PointDataFPM, grid_forward, vtk_write, numpy_to_vtk


def parse_arguments():
    parser = argparse.ArgumentParser(description='CNNP')
    # model parameters
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='input batch size for training (default: 128)')
    parser.add_argument('--sample-size', type=int, default=15000, 
                        help='sample size per file')
    parser.add_argument('--sample-mode', type=str, default='weighted', 
                        help='Sample mode for training data')
    parser.add_argument('-v', '--latent_dimension', dest='latent_dimension', type=int, default=256, 
                        help='The dimension for the pivots')
    parser.add_argument('-r', '--resolution', dest='resolution', type=int, default=16, 
                        help='Save resolution for pivot points')
    parser.add_argument('-p', '--pivot_mode', dest='pivot_mode', type=str, default='regular', 
                        help='Mode used in pivot choosing')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, 
                        help='learning-rate')
    # control parameters
    parser.add_argument('-w', '--load', dest='load', type=str, default=False, 
                        help='load file model')
    parser.add_argument("-i", action='store_true', dest="inference", default=False, 
                        help='Inference mode')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=7, 
                        help='how many batches to wait before logging training status')
    parser.add_argument("--result-dir", dest="result_dir", type=str, default="states", 
                        help='the directory to save the result')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    return args


def inference(args,loader):
        model.eval()
        acc_loss = 0
        print("number of samples: ",len(pd))
        output = torch.zeros((len(pd),4),dtype=float,device="cpu")
        with torch.no_grad():
            for i, d in enumerate(loader):
                d = d.float().to(device)
                output[i*args.batch_size:(i+1)*args.batch_size,:3] = d[:,:3]
                coord = d[:,:3]
                concentration = d[:,3][:,None]
                pred = grid_forward(model,lat_vec,coord)
                output[i*args.batch_size:(i+1)*args.batch_size,3] = pred[:,0]

                MSELoss = nn.MSELoss()(pred,concentration)
                # lat_norm = torch.mean(torch.sum(code ** 2,dim=1))
                loss = MSELoss
                acc_loss += loss.item()
                
                if i % args.log_interval == 0:
                    print('Test: [{:.0f}%]\tLoss: {:.6f}'.format(
                        100. * i / len(loader),
                        loss.item(),
                        ))
                if i == len(loader)-1:
                    print('====> Average loss: {:.4f}'.format(
                        acc_loss / len(loader)))
        output[:,3] *= 357.19
        output[:,:3] *= 10
        output[:,:2] -= 5
        array_dict = {
                "concentration": output[:,3],
            }
        vtk_data = numpy_to_vtk(output[:,:3],array_dict)
        vtk_write(vtk_data,"recon_01_025.vtu")

def train(epoch,args,loader):
        model.train()
        train_loss = 0
        print("number of samples: ",len(pd))
        for i, d in enumerate(loader):
            optimizer.zero_grad()
            d = d.float().to(device)
            coord = d[:,:3]
            concentration = d[:,3][:,None]
            pred = grid_forward(model,lat_vec,coord)
            MSELoss = nn.MSELoss()(pred,concentration)
            # lat_norm = torch.mean(torch.sum(code ** 2,dim=1))
            loss = MSELoss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if i % args.log_interval == 0:
                print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                    epoch, 
                    100. * i / len(loader),
                    loss.item(),
                    ))
            if i == len(loader)-1:
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                    epoch, train_loss / len(loader)))
        if not os.path.isdir(args.result_dir):
            os.mkdir(args.result_dir)
        save_dict = {
            "state": model.state_dict(),
            "config":args,
        }
        torch.save(save_dict,args.result_dir+'/CP{}.pth'.format(epoch))
        print('Checkpoint {} saved !'.format(epoch))


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    try:
        data_path = os.environ['data']
    except KeyError:
        data_path = './data/'

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    res = args.resolution
    lat_dim = args.latent_dimension

    model = AD_SDF(lat_dim).to(device)
    if args.pivot_mode == 'regular':
        # initialize the regular grid latent
        lat_vec = nn.Parameter(torch.empty((lat_dim,res,res,res),requires_grad=True,device=device))
        nn.init.normal_(lat_vec, 0.0 ,1/math.sqrt(lat_dim))
        model.register_parameter("latent_vectors",lat_vec)

    if args.load:
        state_dict = torch.load(args.load)
        state = state_dict['state']
        config = state_dict['config']
        # args = config
        model.load_state_dict(state)
        print('Model loaded from {}'.format(args.load))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if not args.inference:
        for epoch in range(args.epochs):
            file_name = "F:/0.20/run01/025.vtu"
        #     for i,f in enumerate(file_list):
        #         print("file in process: ",f)
        #         print("file processed {}/{}".format(i,len(file_list)))
            pd = PointDataFPM(file_name)
            if args.sample_mode == 'weighted': 
                sampler = WeightedRandomSampler(pd.weights,args.sample_size,replacement=False)
            elif args.sample_mode == 'random': 
                sampler = RandomSampler(pd,replacement=True,num_samples=args.sample_size)
            loader = DataLoader(pd, batch_size=args.batch_size, sampler=sampler)
            train(epoch,args,loader)
    else:
        file_name = "F:/0.20/run01/025.vtu"
        pd = PointDataFPM(file_name)
        loader = DataLoader(pd, batch_size=args.batch_size, shuffle=False, drop_last=False)
        inference(args,loader)


