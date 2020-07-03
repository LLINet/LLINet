from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time

from torch.utils.data.dataloader import default_collate


import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt


import os
import sys
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def get_imgs(img_path, transform=None, normalize=None, seg=False):

    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    if seg == True:
        img = np.array(img.resize((16,16)))
    return img

def get_audios(input_file,args):
    
    y, sr = librosa.load(input_file, sr=None)

    if args.add_noise:
        noise = np.random.randn(len(y))
        y = y + 0.01 * noise
    ws = int(sr * 0.001 * args.audio_window_size)
    st = int(sr * 0.001 * args.audio_stride)
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = args.audio_input_dim, n_fft=ws, hop_length=st)
    feat = np.log(feat + 1e-6)

    feat = [feat]

    feat = np.concatenate(feat, axis=0)
    if args.cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
    return np.swapaxes(feat, 0, 1).astype('float32') 

def pad_collate_train(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')
    for elem in batch:        
        imgs,imgs2, audios, cls_id = elem
        max_input_len = max_input_len if max_input_len > audios.shape[0] else audios.shape[0]       

    for i, elem in enumerate(batch):
        imgs,img2, audios, cls_id = elem
        input_length = audios.shape[1]
        input_dim = audios.shape[0]
        # print('f.shape: ' + str(f.shape))
        feature = np.zeros((max_input_len, input_length), dtype=np.float)
        feature[:audios.shape[0], :audios.shape[1]] = audios       
        
        batch[i] = (imgs,imgs2, feature, cls_id, input_dim)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))
        batch.sort(key=lambda x: x[-1], reverse=True)

    return default_collate(batch)

def pad_collate_t(batch):

    max_input_len = float('-inf')
    max_target_len = float('-inf')
    for elem in batch:
        imgs, audios,cls_id = elem
        max_input_len = max_input_len if max_input_len > audios.shape[0] else audios.shape[0]

    for i, elem in enumerate(batch):
        imgs, audios,cls_id = elem
        input_length = audios.shape[0]
        input_dim = audios.shape[1]
        # print('f.shape: ' + str(f.shape))
        feature = np.zeros((max_input_len, input_dim), dtype=np.float)
        feature[:audios.shape[0], :audios.shape[1]] = audios

        batch[i] = (imgs, feature, cls_id, input_length)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))
        batch.sort(key=lambda x: x[-1], reverse=True)

    return default_collate(batch)

def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')
    for elem in batch:        
        imgs,img_name, audios,seg,cls_id = elem
        max_input_len = max_input_len if max_input_len > audios.shape[0] else audios.shape[0]       

    for i, elem in enumerate(batch):
        imgs,img_name, audios,seg,cls_id = elem
        input_length = audios.shape[0]
        input_dim = audios.shape[1]
        # print('f.shape: ' + str(f.shape))
        feature = np.zeros((max_input_len, input_dim), dtype=np.float)
        feature[:audios.shape[0], :audios.shape[1]] = audios       
        
        batch[i] = (imgs,img_name, feature,seg, cls_id, input_length)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))
        batch.sort(key=lambda x: x[-1], reverse=True)

    return default_collate(batch)


class AVDATA(data.Dataset):
    def __init__(self, data_dir, args, split='train',            
                 transform=None, target_transform=None):
        self.args  = args
        self.split = split
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.data_dir = data_dir

        if split == 'train':
            self.filenames = self.load_filenames(data_dir,'train_image_files.pickle')
            self.class_id = self.load_filenames(data_dir,'train_class_ids.pickle')
            self.audios = self.load_filenames(data_dir,'train_audio_files.pickle')

        else:
            self.filenames = self.load_filenames(data_dir,'test_image_files.pickle')
            self.class_id = self.load_filenames(data_dir,'test_class_ids.pickle')
            self.audios = self.load_filenames(data_dir,'test_audio_files.pickle')

        self.number_example = len(self.filenames)

    def load_filenames(self, data_dir, file_name):
        filepath = os.path.join(data_dir,file_name)
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)   
        return filenames


    def __getitem__(self, index):
        #
        # start = time.time()
        if self.split == 'train':
            key = self.filenames[index]
            cls_id = self.class_id[index]
            cls_name =  key.split('/')[1]       
            data_dir = self.data_dir        
            img_name = '%s/%s' % (data_dir, key)
            img = get_imgs(img_name,self.transform, normalize=self.norm)
            audios_list = self.audios[cls_name]
            audios_num = len(audios_list)
            indx_aud = np.random.randint(0,audios_num)   
            audio_key = audios_list[indx_aud]
            audio_name = '%s/%s' % (data_dir, audio_key)   
            audio_name = audio_name[:-4] + '.wav'
            audio = get_audios(audio_name,self.args)  
            
            return img, audio, cls_id 
        
        else:
            key = self.filenames[index] 
            cls_id = self.class_id[index]
            cls_name =  key.split('/')[1]       
            data_dir = self.data_dir        
            img_name = '%s/%s' % (data_dir, key)
            img = get_imgs(img_name,self.transform, normalize=self.norm)
            audio_key = self.audios[index]
            audio_name = '%s/%s' % (data_dir, audio_key)
            audio_name = audio_name[:-4] + '.wav'
            audio = get_audios(audio_name,self.args)  
            
            seg_name = img_name.replace('video','segment')
            seg = get_imgs(seg_name,None,None,True)
            return img,img_name, audio, seg, cls_id 

    def __len__(self):
        return len(self.filenames)
