import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torchvision import models

class Resnet101(nn.Module):
    def __init__(self,args):
        super(Resnet101, self).__init__()
        self.args = args
        model = models.resnet101(pretrained=True)
        model2 = models.resnet101(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        for param in model2.parameters():
            param.requires_grad = True
        self.define_module(model,model2)
        self.init_trainable_weights()
        self.attention(model)
        for param in self.embedding.parameters():
            param.requires_grad = True

    def define_module(self, model,model2):
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.embedding = model2.layer3
        
        self.fc1 = nn.Linear(2048,2048)
        self.fc2 = nn.Linear(2048,1024)

    def init_trainable_weights(self):
        initrange = 0.1   
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)

    def attention(self,model):
        self.att_conv1 = nn.Conv2d(256,128,(1,1),(1,1))
        self.att_conv2 = nn.Conv2d(128,128,(3,3),(2,2),(1,1))
        self.att_conv3 = nn.Conv2d(128,512,(1,1),(1,1))

    def similarity_mask(self,x,y):
        xdim1, xdim2, xdim3, xdim4 = x.shape
        x = x.view((xdim1, xdim2,-1))
        x = x / torch.norm(x,p=2,keepdim=True)
        y = y / torch.norm(y,p=2,keepdim=True)
        y = y.unsqueeze(1)
        mask = torch.bmm(y, x)
        mask = mask.view((xdim1,-1,xdim3,xdim4))
        e = 1e-7
        mask = torch.clamp(mask,e, 1. - e)
        return mask

    def forward(self, x):
        x = nn.functional.interpolate(x,size=(244, 244), mode='bilinear', align_corners=False)    # (3, 244, 244)
        x = self.conv1(x)    # (64, 122, 122)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)     #(256, 61, 61)
        x = self.layer2(x)     #(512, 31, 31)
       
        features = x
        features = self.embedding(features)
        sig_features = features.view(features.size(0),features.size(1),-1)
        sig_features = F.softmax(sig_features)
        sig_features = sig_features.view(features.size(0),features.size(1),features.size(2),-1)

        x = self.layer3(x)        #(1024, 16, 16)
        x = x.mul(sig_features + 1.0)
        x = self.layer4(x)        #(2048, 8, 8)
        x = self.avgpool(x)       #(2048, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = nn.functional.normalize(x, p=2, dim=1) 
        # features = nn.functional.normalize(features,p=2,dim=1)

        return x, features

