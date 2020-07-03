import os 
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
from .util import *
from .kNN import *
import matplotlib.pyplot as plt 
from matplotlib import cm
from steps import eva
import pandas as pd



def compute_acc(test_image,test_video,cls_idx,test_labels):
    outpred = []
    for i in range(len(test_image)):
        outputlabel = kNNClassify(test_image[i,:],test_video,test_labels,1)
        outpred.append(outputlabel)
    outpred = np.array(outpred)
    acc = np.equal(cls_idx,outpred).mean()
    return acc
 

def train(image_model, audio_model, trans_model,class_model,discr_model,train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir
    save_path = os.path.join(exp_dir, args.result_file)
    save_model_dir = os.path.join(exp_dir,'models')
    if args.Loss_adver:
        aud_labels = Variable(torch.FloatTensor(args.batch_size).fill_(1)).to(device)
        img_labels = Variable(torch.FloatTensor(args.batch_size).fill_(0)).to(device)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)
       
    epoch = 0
    best_acc = 0
    acc_hist = []
    if epoch != 0:        
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))        
        trans_model.load_state_dict(torch.load("%s/models/trans_model.%d.pth" % (exp_dir, epoch)))
        class_model.load_state_dict(torch.load("%s/models/class_model.%d.pth" % (exp_dir, epoch)))
        discr_model.load_state_dict(torch.load("%s/models/discr_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)
    
    image_model = image_model.to(device)
    audio_model = audio_model.to(device)    
    trans_model = trans_model.to(device)
    class_model = class_model.to(device)
    discr_model = discr_model.to(device)
    # Set up the optimizer
    image_trainables = [p for p in image_model.parameters() if p.requires_grad] # if p.requires_grad
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    trans_trainables = [p for p in trans_model.parameters() if p.requires_grad]
    class_trainables = [p for p in class_model.parameters() if p.requires_grad]
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(image_trainables, args.lr_I,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizer_audio = torch.optim.SGD(audio_trainables, args.lr_A,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizer_trans = torch.optim.SGD(trans_trainables, args.lr_T,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr_A,
                                    weight_decay=args.weight_decay,
                                    betas=(0.95, 0.999))
        optimizer_discr = torch.optim.Adam(discr_trainables, args.lr_A,
                                    weight_decay=args.weight_decay,
                                    betas=(0.95, 0.999))

    
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")      

    criterion_hinge = nn.TripletMarginLoss(margin=1.0,p=2)
    criterion_e = nn.MSELoss()
    criterion_s = nn.CosineSimilarity()
    criterion_c = nn.CrossEntropyLoss()    
    criterion_k = nn.KLDivLoss()
    criterion_a = nn.BCELoss()
    best_epoch = 0
    bestmAP = [0,0,0]
    bestRP = [0,0,0]
    best = [0,0,0,0,0]
    att_best = [0,0,0,0]
    while epoch<=args.n_epochs:
        print(epoch)
        epoch += 1
        adjust_learning_rate(args.lr_A, args.lr_decay, optimizer, epoch)
        #adjust_learning_rate(args.lr_A, args.lr_decay, optimizer_discr, epoch)
        end_time = time.time()
        image_model.train()
        audio_model.train()
        trans_model.train()
        class_model.train()

        for i, (image_input, audio_input, cls_id,input_length) in enumerate(train_loader):   
            audio_input = audio_input.float().to(device)
            B = audio_input.size(0)         
            image_input = image_input.float().to(device)
            image_input = image_input.squeeze(1)
            input_length = input_length.float().to(device)            
            
            optimizer.zero_grad()
            optimizer_trans.zero_grad()
            optimizer_audio.zero_grad()
           
            
            audio_output = audio_model(audio_input,input_length)
            audio_att  = trans_model(audio_output)
            image_output, image_features = image_model(image_input)


            loss = 0
            if args.mode == 'sl':
                weight_image_features, att_maps,att = attention(image_features,audio_att,args)  
                lossa1 = batch_loss(weight_image_features,audio_att,cls_id,args)
                loss += lossa1 * args.gamma_att
           
            if args.Loss_batch:
                lossb1 = batch_loss(image_output,audio_output,cls_id,args)

                loss += lossb1 * args.gamma_batch        
                        

            loss.backward()
            optimizer.step()   
            optimizer_trans.step()
            optimizer_audio.step()
            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            
            if i % 40 == 0:                
                print('iteration = %d | loss = %f '%(i,loss))
        
        #retrieval evaluate 
        if args.mode == 'retrieval' and args.eva == 'eva':
            mAPa2i,mAPi2a,A_r1,I_r1,S = eva.validate(audio_model,image_model,test_loader)
            if bestmAP[1]+bestmAP[2] < mAPa2i + mAPi2a:
                bestmAP = [epoch,mAPa2i,mAPi2a]
  
            if bestRP[1]+bestRP[2] < A_r1 + I_r1:
                bestRP = [epoch,A_r1,I_r1]
            if best[1] + best[2] + best[3] + best[4] < mAPa2i + mAPi2a + A_r1 + I_r1:
                best = [epoch,mAPa2i,mAPi2a,A_r1,I_r1]
            info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f}  mAP_A2I: {mAP1_:.4f}  mAP_I2A: {mAP2_:.4f} RP_A2I: {RP1_:.4f} RP_I2A: {RP2_:.4f} Best_mAP:{mAP} Best_RP:{RP} Best:{b}\n \
                '.format(epoch,loss_meter=loss_meter,mAP1_=mAPa2i,mAP2_=mAPi2a,RP1_=A_r1,RP2_=I_r1,mAP=bestmAP,RP=bestRP,b=best)
           
            print (info)
            
            with open(save_path, "a") as file:
                file.write(info)

        #zsl evaluate
        
        if args.mode == 'zsl' and args.eva == 'eva':
            eva.validate_zsl(audio_model,image_model,test_loader)

        
        #sound localization evaluate
        if args.mode == 'sl' and args.eva == 'eva':
            eva.validate_sl(epoch, audio_model,image_model,trans_model,test_loader,args)
