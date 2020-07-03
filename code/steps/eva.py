import torch 
import numpy as np
import torch.nn as nn
import time
from .kNN import *
from .util import *


def compute_acc(test_image,test_video,cls_idx,test_labels):
    outpred = []
    for i in range(len(test_image)):
        outputlabel = kNNClassify(test_image[i,:],test_video,test_labels,1)
        outpred.append(outputlabel)
    outpred = np.array(outpred)
    acc = np.equal(cls_idx,outpred).mean()
    return acc

def normalizeFeature(x):
    # x = d x N dims (d: feature dimension, N: the number of features)
    x = x.cpu().data.numpy()
    x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide
    feature_norm = np.sum(x ** 2, axis=1) ** 0.5# l2-norm
    feat = x / feature_norm[:,np.newaxis]
    return feat

class AverageMeter(object):
    """Computes and stores the average and current value"""

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

def validate(audio_model, image_model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    audio_model = audio_model.to(device)
    audio_model.eval()  
    
    image_model = image_model.to(device)
    image_model.eval()


    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    frame_counts = []
    class_ids = []
    with torch.no_grad():
        test_image = []
        test_audio = []
        labels = []      
        for i, (image_input,image_name, audio_input,segment, cls_id, input_length) in enumerate(val_loader,):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            image_input = image_input.squeeze(1)

            audio_input = audio_input.float().to(device)
            image_input = image_input.float().to(device)
            input_length = input_length.float().to(device)


            audio_output = audio_model(audio_input,input_length)
            image_output,_ = image_model(image_input)               


            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)       
            class_ids.append(cls_id)     
            
            batch_time.update(time.time() - end)
            end = time.time()

            test_image.extend(image_output.cpu().data.numpy())
            test_audio.extend(audio_output.cpu().data.numpy())
            labels.extend(cls_id.cpu().data.numpy())

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        cls_id = torch.cat(class_ids)

   
        mAPi2a  = calc_mAP(image_output,audio_output,cls_id)
        mAPa2i = calc_mAP(audio_output,image_output,cls_id)
        recalls,S = calc_recalls(image_output, audio_output,cls_id)     

        A_r1 = recalls['A_r1']
        I_r1 = recalls['I_r1']

    return mAPa2i,mAPi2a,A_r1,I_r1,S


def validate_zsl(audio_model, image_model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bestacc = 0
    image_model.eval()
    audio_model.eval()
    test_image = []
    test_audio = []
    labels = []
    for i, (image_input,image_name, audio_input,segment, cls_id,input_length) in enumerate(test_loader):
        audio_input = audio_input.float().to(device)
        B = audio_input.size(0)         
        image_input = image_input.float().to(device)
        image_input = image_input.squeeze(1)
        input_length = input_length.float().to(device)            
                   
        audio_output = audio_model(audio_input,input_length)
        image_output, image_features = image_model(image_input)     

        test_image.extend(image_output.cpu().data.numpy())
        test_audio.extend(audio_output.cpu().data.numpy())
        labels.extend(cls_id.cpu().data.numpy())

    test_image = np.array(test_image)
    test_audio = np.array(test_audio)
    labels = np.array(labels)

    all_classes = sorted(np.unique(labels))
    standard_audio = dict.fromkeys(all_classes)
    for i in range(len(labels)):
        standard_audio[labels[i]] = []
    for i in range(len(labels)):
        standard_audio[labels[i]].append(test_audio[i])
    for key in standard_audio.keys():
        standard_audio[key] = np.mean(np.array(standard_audio[key]),axis=0)
        standard_audio[key] = np.array(standard_audio[key])
    sample_labels = np.array([x for x in standard_audio.keys()])
    sample_audio = np.array([x for x in standard_audio.values()])

    acc = compute_acc(test_image,sample_audio,labels,sample_labels)
    if bestacc < acc:
        bestacc = acc
    print('bestacc:',bestacc)

def validate_sl(epoch, audio_model, image_model,trans_model, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model.eval()
    audio_model.eval()
    trans_model.eval()
    att_list = []
    seg_list = []
    labels = []
    feature_list = []
    feature_best = [0,0,0,0,0,0]
    for i, (image_input,image_name, audio_input, segment,cls_id, input_length) in enumerate(test_loader,):
        image_input = image_input.to(device)
        audio_input = audio_input.to(device)
        image_input = image_input.squeeze(1)

        audio_input = audio_input.float().to(device)

        image_input = image_input.float().to(device)
        input_length = input_length.float().to(device)

        audio_output = audio_model(audio_input,input_length)
        audio_att = trans_model(audio_output)
        image_output,image_features = image_model(image_input)               
        #print(image_features.shape,audio_att.shape)
        weight_image_features, att_maps,att = attention(image_features,audio_att,args)


        #image_output = image_output.to('cpu').detach()
        #audio_output = audio_output.to('cpu').detach()
        feature_list.extend(image_features.cpu().data.numpy())
        att_list.extend(att.cpu().data.numpy())
        seg_list.extend(segment.cpu().data.numpy())
        labels.extend(cls_id.cpu().data.numpy())
   
    change = False
    for k in list(range(2,7)):
        iou = cal_iou(att_list,seg_list,k*0.1)
        auc = cal_auc(att_list,seg_list,k*0.1)
        if  feature_best[5] < np.mean(auc):
            feature_best = [epoch,k,auc[20],auc[25],auc[30],np.mean(auc)]
            tmpa = auc
            tmpi = np.mean(iou)
            change = True                
    if change:            
        print(tmpa)
        print(tmpi)
    print(feature_best)
            


def calc_mAP(image_output,audio_output,cls_id):

    value,idx = cls_id.sort()
    image_output = image_output[idx]
    audio_output = audio_output[idx]
    cls_id = cls_id[idx]
    cls_f = -1
    new_cls = []      # classes of the sampled audio
    cls_num = []      #number of each classes of sampled audio
    sampled_audio = []
    i = 0
    j = 0


    for cls_i in cls_id:
        if cls_i!= cls_f:
            new_cls.append(cls_i.unsqueeze(0))  
            sampled_audio.append(audio_output[i].unsqueeze(0))         
            cls_f = cls_i            
            if i!=0:
                cls_num.append(j)    #             
            j = 1 
        else:
            j += 1
        i += 1   

    cls_num.append(j)   


    new_cls = torch.cat(new_cls)
    sampled_audio = torch.cat(sampled_audio)
       
    # using consine similarity
    img_f = torch.from_numpy(normalizeFeature(image_output))
    aud_f = torch.from_numpy(normalizeFeature(sampled_audio) )
    S = aud_f.mm(img_f.t()) 
    value, indx = torch.sort(S,dim=1,descending=True)
    class_sorted = cls_id[indx]
    clss_m2 = new_cls.unsqueeze(-1).repeat(1,S.shape[1])
    
    mask = (class_sorted==clss_m2).bool()
    class_sorted_filed = class_sorted.data.masked_fill_(mask,-10e5)   

    v, index = torch.sort(class_sorted_filed,dim=1)
    index = index +1
    sc = 0.0
    ap = 0.0

    for i in range(index.shape[0]):
        sc = 0.0
        num = cls_num[i]
        for k in range(num):    
            position =  index[i][:num]  
            position = sorted(position)     
            sc += (k+1.0)/(position[k]).float()
        ap += sc/cls_num[i]
    
    mAP = ap/(mask.shape[0])
    return mAP

def calc_recalls(image_outputs, audio_outputs,cls_id):
    """
    Computes recall at 1, 5, and 10 given encoded image and audio outputs.
    """
        
    # S = compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype=simtype)
    image_L2  = normalizeFeature(image_outputs)
    audio_L2  = normalizeFeature(audio_outputs)
    image_L2 = torch.from_numpy(image_L2)
    audio_L2 = torch.from_numpy(audio_L2)
    
    S = image_L2.mm(audio_L2.t())  
    n = S.size(0)
    
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = 0
    I_r1 = 0    
    for i in range(n):
        if cls_id[A2I_ind[0, i]] == cls_id[i]:
            A_r1 += 1
        if cls_id[I2A_ind[i, 0]] == cls_id[i]:
            I_r1 += 1

    A_r1 = A_r1 / n

    I_r1 = I_r1 / n
    
    recalls = {'A_r1':A_r1, 'I_r1':I_r1}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls,S

def cal_iou(att,seg,threshold=0.5):
    num = len(att)
    att = np.array(att)
    seg = np.array(seg)
    intersection = np.zeros(num)
    union = np.zeros(num)
    iou = np.zeros(num)
    for i in range(num):
        if len(att[i]) == 1024:
            tmpatt = np.mean(att[i],0)
        else:
            tmpatt = att[i]
        tmpatt = np.where(tmpatt > (np.max(tmpatt) * threshold), 1 , 0)
        tmpseg = np.mean(seg[i],axis=2)
        tmpseg = np.where(tmpseg > 0, 1, 0)
        intersection[i] = np.sum(np.where(tmpatt * tmpseg == 1, 1, 0))
        union[i] = np.sum(tmpatt) + np.sum(tmpseg) - intersection[i]
        iou[i] = intersection[i] / (union[i] + 1e-5)
    return iou

def cal_auc(att,seg,la=0.5):
    iou = cal_iou(att,seg,la)
    thresholds = 0.02 * np.array(list(range(51)))
    ind = 0
    auc = np.zeros(len(thresholds))
    for t in thresholds:
        pos = np.sum(np.where(iou >= t,1,0))
        neg = np.sum(np.where(iou < t,1,0))
        auc[ind]= pos / (pos + neg)
        ind += 1
    return auc

def att(audio_model, image_model, trans_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    audio_model = audio_model.to(device)
    audio_model.eval()  
    
    image_model = image_model.to(device)
    image_model.eval()

    trans_model = trans_model.to(device)
    audio_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    frame_counts = []
    with torch.no_grad():
        feature_list = []
        seg_list = []
        labels = []      
        image_file = []
        att_list = []
        for i, (image_input,image_name, audio_input, segment,cls_id, input_length) in enumerate(val_loader,):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            image_input = image_input.squeeze(1)

            audio_input = audio_input.float().to(device)
            
            image_input = image_input.float().to(device)
            input_length = input_length.float().to(device)     
             
           
            audio_output = audio_model(audio_input,input_length)
            audio_att = trans_model(audio_output)
            image_output,image_features = image_model(image_input)               
            #print(image_features.shape,audio_att.shape)
            weight_image_features, att_maps,att = attention(image_features,audio_att,args)

            att_list.extend(att.cpu().data.numpy())
            seg_list.extend(segment.cpu().data.numpy())
            labels.extend(cls_id.cpu().data.numpy())
            image_file.extend(image_name)
        np.save('file.npy',np.array(image_file))
        np.save('att.npy',np.array(att_list))

