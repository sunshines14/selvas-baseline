import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import model.resnet as resnet
import configure as c 
import math
import random
import numpy as np


class custom_model(nn.Module):
    def __init__(self, embedding_size, num_classes, backbone='resnet18'):
        super(custom_model, self).__init__()
        self.backbone = backbone
        if backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=False)
        elif backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False)
        elif backbone == 'resnetxt50_32x4d':
            self.pretrained = resnet.resnext50_32x4d(pretrained=False)
        elif backbone == 'resnext101_32x8d':
            self.pretrained = resnet.resnext101_32x8d(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        
        self.fc1 = nn.Linear(1024, embedding_size) 
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(1024, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.fc3 = nn.Linear(1024, embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.LeakyReLU() 
        self.last = nn.Linear(embedding_size, num_classes)
        self.bn_cat = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        p_0 = F.adaptive_avg_pool2d(x,1)
        p_0 = torch.squeeze(p_0)
        p_0 = p_0.view(x.size(0), -1)
        #print ("p_0:", p_0.size())

        #============= mutiple pooling 1 =============
        x = self.pretrained.layer1(x)
        p_1 = F.adaptive_avg_pool2d(x,1)
        p_1 = torch.squeeze(p_1)
        p_1 = p_1.view(x.size(0), -1)
        #print ("p_1:", p_1.size())

        #==== masked cross self-attention block 1 ==== 
        d_k_1 = p_1.size(1)
        q_1 = p_0.transpose(0,1)
        k_1 = p_1
        v_1 = p_1.transpose(0,1)
        q_1_m = attention(masking(q_1), k_1, d_k_1, v_1)
        
        d_k_1_c = p_0.size(1)
        q_1_c = p_1.transpose(0,1)
        k_1_c = p_0
        v_1_c = p_0.transpose(0,1)
        q_1_m_c = attention(q_1_c, masking(k_1_c), d_k_1_c, v_1_c)
        
        z_1 = torch.matmul(q_1_m, q_1_m_c.transpose(0,1))
        #print ("z1:", z_1.size())

        #============= mutiple pooling 2 =============
        x = self.pretrained.layer2(x)
        p_2 = F.adaptive_avg_pool2d(x,1)
        p_2 = torch.squeeze(p_2)
        p_2 = p_2.view(x.size(0), -1)
        #print ("p_2:", p_2.size())

        #==== masked cross self-attention block 2 =====
        d_k_2 = p_2.size(1)
        q_2 = p_1.transpose(0,1)
        k_2 = p_2
        v_2 = p_2.transpose(0,1)
        q_2_m = attention(masking(q_2), k_2, d_k_2, v_2)
        
        d_k_2_c = p_1.size(1)
        q_2_c = p_2.transpose(0,1)
        k_2_c = p_1
        v_2_c = p_1.transpose(0,1)
        q_2_m_c = attention(q_2_c, masking(k_2_c), d_k_2_c, v_2_c)
        
        z_2 = torch.matmul(q_2_m, q_2_m_c.transpose(0,1))
        #print ("z2:", z_2.size())

        #============= mutiple pooling 3 =============
        x = self.pretrained.layer3(x)
        p_3 = F.adaptive_avg_pool2d(x,1)
        p_3 = torch.squeeze(p_3)
        p_3 = p_3.view(x.size(0), -1)
        #print ("p_3:", p_3.size())

        #==== masked cross self-attention block 3 ====
        d_k_3 = p_3.size(1)
        q_3 = p_2.transpose(0,1)
        k_3 = p_3
        v_3 = p_3.transpose(0,1)
        q_3_m = attention(masking(q_3), k_3, d_k_3, v_3)
        
        d_k_3_c = p_2.size(1)
        q_3_c = p_3.transpose(0,1)
        k_3_c = p_2
        v_3_c = p_2.transpose(0,1)
        q_3_m_c = attention(q_3_c, masking(k_3_c), d_k_3_c, v_3_c)
        
        z_3 = torch.matmul(q_3_m, q_3_m_c.transpose(0,1))
        #print ("z3:", z_3.size())

        #============= mutiple pooling 4 =============
        x = self.pretrained.layer4(x)
        p_4 = F.adaptive_avg_pool2d(x,1)
        p_4 = torch.squeeze(p_4) 
        p_4 = p_4.view(x.size(0), -1)
        #print ("p_4:", p_4.size())

        #==== masked cross self-attention block 4 ====
        d_k_4 = p_4.size(1)
        q_4 = p_3.transpose(0,1)
        k_4 = p_4
        v_4 = p_4.transpose(0,1)
        q_4_m = attention(masking(q_4), k_4, d_k_4, v_4)
        
        d_k_4_c = p_3.size(1)
        q_4_c = p_4.transpose(0,1)
        k_4_c = p_3
        v_4_c = p_3.transpose(0,1)
        q_4_m_c = attention(q_4_c, masking(k_4_c), d_k_4_c, v_4_c)
        
        z_4 = torch.matmul(q_4_m, q_4_m_c.transpose(0,1))
        #print ("z4:", z_4.size())

        #========== self-attention matrix ===========
        z_1 = torch.matmul(p_0, z_1)
        #print (z_1.size())
        z_2 = torch.matmul(z_1, z_2)
        #print (z_2.size())
        z_3 = torch.matmul(z_2, z_3)
        #print (z_3.size())
        z_4 = torch.matmul(z_3, z_4)
        #print (z_4.size())
        #print ("z4:", z_4.size())

        #============== concatenation ===============
        out = torch.cat((z_4, p_4), 1)
        out = self.relu(self.bn_cat(out))
        #print ("total:", out.size())
        
        #========== fully-connected layers ==========
        out = self.fc1(out)
        out = self.relu(self.bn1(out))
        out = self.fc2(out)
        out = self.relu(self.bn2(out))
        out = self.fc3(out) 
        spk_embedding = out 
        #print ("embedding:", spk_embedding.size())
        out = self.relu(self.bn3(out))
        out = self.last(out)
        out = F.log_softmax(out, dim=-1)
        
        return spk_embedding, out


def attention(q, k, d_k, v):
    #scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = torch.matmul(q, k) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    #if dropout is not None:
        #scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

def masking(target):
    #k is scaling vector for masking
    mask = np.triu(np.ones((target.size(0),target.size(1))), k=0)
    tmp = mask.ravel()
    np.random.shuffle(tmp)
    ran_mask = tmp.reshape((target.size(0),target.size(1)))
    ran_mask = torch.from_numpy(ran_mask).float().cuda()
    output = torch.mul(target, ran_mask)
    return output


from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = custom_model(c.EMBEDDING_SIZE, c.N_CLASSES).to(device)
summary(model, input_size=(1, c.FILTER_BANK, c.NUM_WIN_SIZE))
