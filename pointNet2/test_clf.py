import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader, load_data
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils import test, save_checkpoint
from model.pointnet2 import PointNet2ClsMsg
from model.pointnet import PointNetCls, feature_transform_reguliarzer



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=10, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
   # parser.add_argument('--pretrain', type=str, default='./experiment/checkpoints/pointnet-0.895682-0187.pth',help='whether use pretrain model')
    parser.add_argument('--pretrain', type=str, default='./pointnet2-0.921761-0130.pth',help='whether use pretrain model')
    parser.add_argument('--rotation',  default=None, help='range of training rotation')
    parser.add_argument('--model_name', default='pointnet2', help='range of training rotation')
    parser.add_argument('--feature_transform', default=False, help="use feature transform in pointnet")
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    datapath = './data/ModelNet/'
    train_data, train_label, test_data, test_label = load_data(datapath, classification=True)

    if args.rotation is not None:
        ROTATION = (int(args.rotation[0:2]),int(args.rotation[3:5]))
    else:
        ROTATION = None

    if ROTATION is not None:
        print('The range of training rotation is',ROTATION)
    testDataset = ModelNetDataLoader(test_data, test_label, rotation=ROTATION)
  
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False)

    '''MODEL LOADING'''
    num_class = 40
    classifier = PointNetCls(num_class,args.feature_transform).cuda() if args.model_name == 'pointnet' else PointNet2ClsMsg().cuda()
    if args.pretrain is not None:
        print('Use pretrain model...')
       
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    with torch.no_grad():
        acc = test(classifier.eval(), testDataLoader)
        print('\r Test Accuracy: %f' % acc)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
