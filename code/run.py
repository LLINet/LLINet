import argparse
import os
import pickle
import sys
import time
import ast
import torch
import random
import datetime
import pprint
import dateutil.tz
import numpy as np
from PIL import Image
from dataloaders.dataset import AVDATA, pad_collate_t, pad_collate, pad_collate_train
from models import  ImageModels, AudioModels,classification
from steps import train
import torchvision.transforms as transforms 
import torch.nn as nn
from steps import eva

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#'/media/shawn/data/Data/birds'
#'/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/birds'
parser.add_argument('--data_path', type = str, default='../Dataset_32classes/traindata') #
parser.add_argument('--class_num',type = int, default= 24)
parser.add_argument('--exp_dir', type = str, default= 'outputs/baseline')
parser.add_argument('--result_file',type = str, default='Baseline_class.text')
parser.add_argument("--resume", action="store_true", default=True,
        help="load from exp_dir if True")
parser.add_argument("--optim", type=str, default="sgd",
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument("--mode", type=str, default="retrieval",
        help="training task", choices=["retrieval", "zsl","sl"])
parser.add_argument("--eva", type=str, default="eva",
        help="whether evaluate", choices=["eva", "no"])
parser.add_argument('--batch_size', '--batch_size', default=32, type=int,
    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--workers',default=0,type=int,help='number of worker in the dataloader')
parser.add_argument('--lr_A', '--learning-rate-attribute', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_T', '--learning-rate-Trans', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_I', '--learning-rate-image', default=0.001, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=10, type=int, metavar='LRDECAY',
    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
    metavar='W', help='weight decay (default: 1e-4)')     #1e-3
parser.add_argument("--n_epochs", type=int, default=100,
        help="number of maximum training epochs")
parser.add_argument('--CUDA',default=True,help='whether use GPU')
parser.add_argument('--gpu_id',type = int, default= 0)
parser.add_argument('--manualSeed',type=int,default= 200, help='manual seed')
parser.add_argument('--img_size',type=int,default = 244,help='image size')


parser.add_argument('--audio_window_size',type=int,default=25)
parser.add_argument('--audio_stride',type=int, default=10)
parser.add_argument('--audio_input_dim',type=int, default=40)
parser.add_argument('--cmvn',default=False)
parser.add_argument('--add_noise',default=False)

parser.add_argument('--rnn_type',type=str,default='GRU')

parser.add_argument('--audio_self_att',default=True)
parser.add_argument('--image_self_att',default=True)

parser.add_argument('--Loss_cont',default = False)
parser.add_argument('--gamma_cont',type=float,default = 1.0)
parser.add_argument('--Loss_batch',type=ast.literal_eval,default = True)
parser.add_argument('--gamma_batch',type=float,default = 1.0)
parser.add_argument('--Loss_dist',default = False)
parser.add_argument('--gamma_dist',type=float,default = 1.0)
parser.add_argument('--Loss_hinge',default = False)
parser.add_argument('--gamma_hinge',type=float,default = 1.0)
parser.add_argument('--Loss_clss',default=False)
parser.add_argument('--gamma_clss',type=float,default=1.0)
parser.add_argument('--Loss_adver',type=ast.literal_eval,default=False)
parser.add_argument('--ad',type=float,default=1.0)
parser.add_argument('--gamma_att',type=float,default=1.0)
# parser.add_argument()

parser.add_argument('--smooth_gamma1',type=float,default=1)
parser.add_argument('--smooth_gamma2',type=float,default=5)
parser.add_argument('--smooth_gamma3',type=float,default=10)


args = parser.parse_args()

resume = args.resume

print(args)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed(args.manualSeed)
#torch.set_num_threads(4)
if args.CUDA:    
    torch.cuda.manual_seed(args.manualSeed)  
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):   
    np.random.seed(args.manualSeed + worker_id)


imsize = args.img_size
image_transform = transforms.Compose([
    transforms.Resize(int(imsize * 76 / 64)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])

test_image_transform = transforms.Compose([
    transforms.Resize(imsize),
    transforms.CenterCrop(imsize)])

dataset = AVDATA(args.data_path, args,'train',
                        transform=image_transform)
dataset_test = AVDATA(args.data_path, args,'test',transform=test_image_transform)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size,
    drop_last=True, shuffle=True,num_workers=args.workers,collate_fn=pad_collate_t,worker_init_fn=worker_init_fn)
val_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size,
    drop_last=True, shuffle=False,num_workers=args.workers,collate_fn=pad_collate)

image_model = ImageModels.Resnet101(args)
audio_model = AudioModels.Resnet(args)
trans_model = AudioModels.Linear_transform(args)
class_model = classification.CLASSIFIER(args)
discr_model = classification.DISCRIMINATOR(args)


train(image_model, audio_model,trans_model,class_model,discr_model,train_loader, val_loader, args)
'''
aweight = torch.load('asl_audio.pth')
iweight = torch.load('asl_image.pth')
tweight = torch.load('asl_trans.pth')
audio_model.load_state_dict(aweight)
image_model.load_state_dict(iweight)
trans_model.load_state_dict(tweight)

eva.att(audio_model,image_model,trans_model,val_loader,args)
'''
