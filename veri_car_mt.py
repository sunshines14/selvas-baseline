import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
import os
import configure as c
import numpy as np
import pickle
from torch.autograd import Variable
from DB_wav_reader import read_test_feats_structure
from SR_Dataset import read_MFB, ToTensorTestInput
from model.model import custom_model
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from collections import OrderedDict


def load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes):
    model = custom_model(embedding_size=embedding_size, num_classes=n_classes)
    if use_cuda:
       model.cuda()
    print('=> loading checkpoint')
    checkpoint = torch.load(log_dir + '/checkpoint_' + str(cp_num) + '.pth')
    #model.load_state_dict(checkpoint['state_dict'])
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v 
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def get_test_DB(dataroot_dir):
    test_DB = read_test_feats_structure(dataroot_dir)
    test_DB = test_DB.reset_index(drop=True)
    return test_DB

def get_embeddings(use_cuda, filename, model, test_frames):
    input, label = read_MFB(filename) 
    activation = 0 
    
    # ===== 1 =====
    tot_segments = math.ceil(len(input)/test_frames) 
    with torch.no_grad():
        for i in range(tot_segments):
            temp_input = input[i*test_frames:i*test_frames+test_frames]            
            TT = ToTensorTestInput()
            temp_input = TT(temp_input) 
            if use_cuda:
                temp_input = temp_input.cuda()
            temp_activation,_ = model(temp_input)
            activation += torch.sum(temp_activation, dim=0, keepdim=True)
    
    activation = l2_norm(activation, 1)
    return activation

def l2_norm(input, alpha):
    input_size = input.size()  
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
    output = output * alpha
    return output

def perform_verification(use_cuda, model, enroll_speaker, test_DB, test_path, test_frames, thres):
    enroll_embedding = enroll_per_spk(use_cuda, test_frames, model, test_DB, enroll_speaker, test_path)
    test_embedding = get_embeddings(use_cuda, test_path, model, test_frames)
    score = F.cosine_similarity(test_embedding, enroll_embedding)
    score = score.data.cpu().numpy() 
    if score > thres:
        result = 'Accept'
    else:
        result = 'Reject'
    return score, result

def enroll_per_spk(use_cuda, test_frames, model, DB, enroll_speaker, test_path):
    enroll_speaker_list = DB[DB['speaker_id'].str.contains(enroll_speaker)]
    n_files = len(enroll_speaker_list) 
    embeddings = 0 
    
    for i in range(n_files):
        filename = str(enroll_speaker_list['filename'][i:i+1])
        filename = str((filename.split('    ')[1]).split('\n')[0])
        if filename != test_path :
            activation = get_embeddings(use_cuda, filename, model, test_frames)
            embeddings += activation
    return embeddings

def get_protocol(protocol_file):
    with open(protocol_file, 'r') as f:
        protocol = f.read().splitlines()
        return protocol
           
def main():
    log_dir = c.MODEL_LOAD_DIR
    test_dir = c.TEST_FEAT_DIR
    use_cuda = c.USE_CUDA
    embedding_size = c.EMBEDDING_SIZE
    cp_num = c.CHECK_POINT
    n_classes = c.N_CLASSES
    test_frames = c.TEST_FRAMES

    model = load_model(use_cuda, log_dir, cp_num, embedding_size, n_classes)
    
    test_DB = get_test_DB(c.TEST_FEAT_DIR)
    score_array = np.array([])
    y_array = np.array([])
    protocol = get_protocol(c.PROTOCOL)
    count = 1
    for pair in protocol:
        pair_splited = pair.split(' ')
        file1 = pair_splited[1]
        enroll_path = file1.replace('.wav',c.TEST_FEATURE)
        enroll_speaker = enroll_path.split('/')[-3]
        file2 = pair_splited[2]
        test_path = file2.replace('.wav',c.TEST_FEATURE)
        same = int(pair_splited[0])

        thres = c.THRES
        score, result = perform_verification(use_cuda, model, enroll_speaker, test_DB, test_path, test_frames, thres)
        score_array = np.append(score_array, score)
        if same == 1 :
            y_array = np.append(y_array, 1)
        elif same == 0 :
            y_array = np.append(y_array, 0)
        print (enroll_speaker + "  vs  " + file2)
        print (count, score, same, result)
        count = count + 1

    fpr, tpr, thresholds = metrics.roc_curve(y_array, score_array)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print ('==============================')
    print ('EER: %f' % eer)
    print ('Threshold: %f' % thresh)
    print ('==============================')
    # store result 
    #with open('result/y_array.p', 'wb') as f:
    #    pickle.dump(y_array, f)
    #with open('result/score_array.p', 'wb') as f:
    #    pickle.dump(score_array, f)

if __name__ == '__main__':
    main()
