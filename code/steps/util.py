import math
import pickle
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def normalizeFeature(x):	
    
    x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide\
    feature_norm = torch.sum(x**2, axis=1)**0.5 # l2-norm
    feat = x / feature_norm.unsqueeze(-1)
    return feat


def attention(img_features, audios, args):

    batch_size = args.batch_size
    labels = Variable(torch.LongTensor(range(batch_size)))
    labels = labels.cuda()
    context = img_features
    ih, iw = context.size(2), context.size(2)
    audio_features = audios
    context = context.view(batch_size,-1,ih*iw)
    contextT = context.transpose(1,2)  
    
    audios = audios.unsqueeze(-1)
    att_maps = torch.bmm(contextT,audios)*args.smooth_gamma1
    att = att_maps.view(batch_size,ih,iw)
    att_maps = nn.Softmax(dim=1)(att_maps)        
    weight_feature = torch.bmm(context,att_maps).squeeze(-1) + context.sum(-1)
    att_maps = att_maps.view(batch_size,ih,iw)  

    return weight_feature, att_maps, att   



def batch_loss(cnn_code, rnn_code, class_ids,args,eps=1e-8):

    batch_size = args.batch_size
    labels = Variable(torch.LongTensor(range(batch_size)))
    labels = labels.cuda()   
    
    masks = []
    if class_ids is not None:
        class_ids =  class_ids.data.cpu().numpy()
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)

        masks = torch.ByteTensor(masks)
        masks = masks.to(torch.bool)
        if args.CUDA:
            masks = masks.cuda()


    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)


    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * args.smooth_gamma3


    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0 + loss1



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.5 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

