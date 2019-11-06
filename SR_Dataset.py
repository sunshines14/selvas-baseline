import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import os
import pickle 
import numpy as np
import configure as c
import math
from DB_wav_reader import read_DB_structure


def read_MFB(filename):
    with open(filename, 'rb') as f:
        feat_and_label = pickle.load(f)   
    feature = feat_and_label['feat'] 
    label = feat_and_label['label']
    
    # ===== librosa =====
    if c.USE_TRANSPOSE:
        feature = np.transpose(feature)

    # ===== segment =====
    #start_sec, end_sec = 0.25, 0.25
    #start_frame = int(start_sec / 0.01)
    #end_frame = len(feature) - int(end_sec / 0.01)
    #ori_feat = feature
    #feature = feature[start_frame:end_frame,:]
    #assert len(feature) > c.NUM_WIN_SIZE, ('length is too short. len:%s, ori_len:%s, file:%s' % (len(feature), len(ori_feat), filename))
    return feature, label

class TruncatedInputfromMFB(object):
    def __init__(self, input_per_file=1):
        super(TruncatedInputfromMFB, self).__init__()
        self.input_per_file = input_per_file
    
    def __call__(self, frames_features):
        network_inputs = []
        num_frames = len(frames_features)
        win_size = c.NUM_WIN_SIZE
        
        # ===== 1 =====
        half_win_size = int(win_size/2)
        if num_frames - half_win_size < half_win_size:
            while num_frames - half_win_size <= half_win_size:
                frames_features = np.append(frames_features, frames_features[:num_frames,:], axis=0)
                num_frames =  len(frames_features)
        for i in range(self.input_per_file):
            try:
                j = random.randrange(half_win_size, num_frames - half_win_size)
            except: 
                j = random.randrange(half_win_size, (num_frames - half_win_size)+1)
            if not j:
                frames_slice = np.zeros(num_frames, c.FILTER_BANK, 'float64')
                frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
            else:
                frames_slice = frames_features[j - half_win_size:j + half_win_size]
            network_inputs.append(frames_slice)

        # ===== 2 =====
        #for i in range(self.input_per_file):
        #    fixed_frames_feature = np.zeros(shape=(win_size, c.FILTER_BANK))
        #    for j in range(win_size):
        #        fixed_frames_feature[j] = frames_features[abs(int(num_frames * ((j+1)/win_size))-1)] 
        #    network_inputs.append(fixed_frames_feature)
        
        # ===== 3 =====
        #n = 0
        #for i in range(self.input_per_file):
        #    while num_frames > (win_size*n)+(win_size):
        #        frames_slice = frames_features[win_size*n:(win_size*n)+(win_size)]
        #        network_inputs.append(frames_slice)
        #        n = n+1
        #    return np.array(network_inputs)
        #    frames_slice = frames_features[win_size*n:num_frames]
        #    network_inputs.append(frames_slice)
        #    return np.array(network_inputs)

        # ===== 4 =====
        #network_inputs.append(frames_features)

        # ===== 5 =====
        #dim = c.FILTER_BANK
        #tot_segments = math.ceil(num_frames/win_size)
        #temp_input = torch.ones((win_size, dim)) 
        #with torch.no_grad():
        #    for i in range(tot_segments):
        #        if num_frames > i*win_size+win_size:
        #            temp = frames_features[i*win_size:i*win_size+win_size]
        #            temp = torch.from_numpy(np.array(temp)).float() 
        #            temp_input = torch.mul(temp_input,temp)
        #    temp_input = np.asarray(temp_input)
        #    temp_input = temp_input.reshape(1, win_size, dim)
        #return temp_input
        
        return np.array(network_inputs)

class ToTensorInput(object):
    def __call__(self, np_feature):
        if isinstance(np_feature, np.ndarray):
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() 
            return ten_feature

class ToTensorDevInput(object):
    def __call__(self, np_feature):
        if isinstance(np_feature, np.ndarray):
            np_feature = np.expand_dims(np_feature, axis=0)
            assert np_feature.ndim == 3, 'Data is not a 3D tensor. size:%s' %(np.shape(np_feature),)
            ten_feature = torch.from_numpy(np_feature.transpose((0,2,1))).float() 
            return ten_feature

class ToTensorTestInput(object):
    def __call__(self, np_feature):
        if isinstance(np_feature, np.ndarray):
            np_feature = np.expand_dims(np_feature, axis=0)
            np_feature = np.expand_dims(np_feature, axis=1)
            assert np_feature.ndim == 4, 'Data is not a 4D tensor. size:%s' %(np.shape(np_feature),)
            ten_feature = torch.from_numpy(np_feature.transpose((0,1,3,2))).float() 
            return ten_feature

# ===== 1 =====
def collate_fn_variable(batch):
    batch.sort(key=lambda x: x[0].shape[2], reverse=True)
    #data, target = zip(*batch)
    n = 0
    for i in batch:
        tmp = torch.split(i[0], i[0].size(0), dim=0)
        data = torch.stack(tmp, dim=0)
        print (data.size())
    #    for j in range(len(tmp)):
    #        if j == 0:
    #            data = torch.cat(tuple(tmp[j].unsqueeze(0)), dim=0)
    #        else:
    #            data = torch.cat(data,tuple(tmp[j].unsqueeze(0)), dim=0)
    #    print (data.size())
    #print (data.size())
    target = [i[1] for i in batch]
    target = torch.LongTensor(target)
    return data, target

# ===== 2 =====
def collate_fn_feat_padded(batch):
    batch.sort(key=lambda x: x[0].shape[2], reverse=True)
    feats, labels = zip(*batch) 
    # Merge labels => torch.Size([batch_size,1])
    labels = torch.stack(labels, 0)
    labels = labels.view(-1) 
    # Merge frames
    # in decreasing order
    lengths = [feat.shape[2] for feat in feats] 
    max_length = lengths[0]
    # features_mod.shape => torch.Size([batch_size, n_channel, dim, max(n_win)])
    # convert to FloatTensor (it should be!). torch.Size([batch, 1, feat_dim, max(n_win)])
    padded_features = torch.zeros(len(feats), feats[0].shape[0], feats[0].shape[1], feats[0].shape[2]).float() 
    for i, feat in enumerate(feats):
        end = lengths[i]
        num_frames = feat.shape[2]
        while max_length > num_frames:
            feat = torch.cat((feat, feat[:,:,:end]), 2)
            num_frames = feat.shape[2]   
        padded_features[i, :, :, :] = feat[:,:,:max_length] 
    return padded_features, labels

class DataLoader(data.Dataset):
    def __init__(self, DB, loader, spk_to_idx, transform=None, *arg, **kw):
        self.DB = DB
        self.len = len(DB)
        self.transform = transform
        self.loader = loader
        self.spk_to_idx = spk_to_idx
    
    def __getitem__(self, index):
        feat_path = self.DB['filename'][index]
        feature, label = self.loader(feat_path)
        label = self.spk_to_idx[label]
        label = torch.Tensor([label]).long()
        if self.transform:
            feature = self.transform(feature)
        return feature, label
    
    def __len__(self):
        return self.len 
