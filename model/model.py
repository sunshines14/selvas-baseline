import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import model.resnet as resnet
import configure as c 


class custom_model(nn.Module):
    def __init__(self, embedding_size, num_classes, backbone='resnet34'):
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
            
        self.fc0 = nn.Linear(3904, embedding_size)
        self.bn0 = nn.BatchNorm1d(embedding_size)
        self.fc1 = nn.Linear(1024, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.fc2 = nn.Linear(1024, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.LeakyReLU()
        self.last = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        #x = self.pretrained.maxpool(x)
        p_0 = F.adaptive_avg_pool2d(x,1)
        p_0 = torch.squeeze(p_0)
        p_0 = p_0.view(x.size(0), -1)

        x = self.pretrained.layer1(x)
        p_1 = F.adaptive_avg_pool2d(x,1)
        p_1 = torch.squeeze(p_1)
        p_1 = p_1.view(x.size(0), -1)

        x = self.pretrained.layer2(x)
        p_2 = F.adaptive_avg_pool2d(x,1)
        p_2 = torch.squeeze(p_2)
        p_2 = p_2.view(x.size(0), -1)
                        
        x = self.pretrained.layer3(x)
        p_3 = F.adaptive_avg_pool2d(x,1)
        p_3 = torch.squeeze(p_3)
        p_3 = p_3.view(x.size(0), -1)

        x = self.pretrained.layer4(x)
        p_4 = F.adaptive_avg_pool2d(x,1) 
        p_4 = torch.squeeze(p_4) 
        p_4 = p_4.view(x.size(0), -1)

        p_1 = torch.cat((p_1,p_0),1)
        p_2 = torch.cat((p_2,p_1),1)
        p_3 = torch.cat((p_3,p_2),1)
        out = torch.cat((p_4,p_3),1)

        out = self.fc0(out)
        out = self.relu(self.bn0(out))
        out = self.fc1(out)
        out = self.relu(self.bn1(out))
        out = self.fc2(out)
        spk_embedding = out 
        out = self.relu(self.bn2(out))
        out = self.last(out)
        
        return spk_embedding, out

from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = custom_model(c.EMBEDDING_SIZE, c.N_CLASSES).to(device)
summary(model, input_size=(1, c.FILTER_BANK, c.NUM_WIN_SIZE))
