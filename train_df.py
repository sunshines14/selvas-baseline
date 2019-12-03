import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time
import os
import numpy as np
import configure as c
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from DB_wav_reader import read_train_feats_structure, read_train_feats_structure_SA
from SR_Dataset import read_MFB, TruncatedInputfromMFB, ToTensorInput, ToTensorDevInput, DataLoader, collate_fn_variable, collate_fn_feat_padded
from model.model import custom_model
from utils.pytorchtools import EarlyStopping


def load_dataset(val_ratio):
    train_DB, valid_DB = split_train_dev(c.TRAIN_FEAT_DIR, val_ratio) 
    file_loader = read_MFB 
    transform = transforms.Compose([
        TruncatedInputfromMFB(), 
        ToTensorInput() 
    ])
    transform_T = ToTensorDevInput()
    speaker_list = sorted(set(train_DB['speaker_id'])) 
    spk_to_idx = {spk: i for i, spk in enumerate(speaker_list)}
    train_dataset = DataLoader(DB=train_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx)
    valid_dataset = DataLoader(DB=valid_DB, loader=file_loader, transform=transform_T, spk_to_idx=spk_to_idx) 
    n_classes = len(speaker_list)     
    return train_dataset, valid_dataset, n_classes

def split_train_dev(train_feat_dir, valid_ratio):
    train_valid_DB = read_train_feats_structure_SA(train_feat_dir)
    #train_valid_DB = read_train_feats_structure(train_feat_dir)
    total_len = len(train_valid_DB) 
    valid_len = int(total_len * valid_ratio/100.)
    train_len = total_len - valid_len
    shuffled_train_valid_DB = train_valid_DB.sample(frac=1).reset_index(drop=True)
    train_DB = shuffled_train_valid_DB.iloc[:train_len]
    valid_DB = shuffled_train_valid_DB.iloc[train_len:]
    train_DB = train_DB.reset_index(drop=True)
    valid_DB = valid_DB.reset_index(drop=True)
    print('\nTraining set %d utts (%0.1f%%)' %(train_len, (train_len/total_len)*100))
    print('Validation set %d utts (%0.1f%%)' %(valid_len, (valid_len/total_len)*100))
    print('Total %d utts' %(total_len)) 
    return train_DB, valid_DB

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

def create_optimizer(optimizer, model, new_lr, wd):
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0,
                              weight_decay=wd)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=wd)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=new_lr,
                                  weight_decay=wd)
    return optimizer

def visualize_the_losses(train_loss, valid_loss):
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss, label='Validation Loss')
    
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 3.5) 
    plt.xlim(0, len(train_loss)+1) 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot_saved/loss_plot.png', bbox_inches='tight')

def train(train_loader, model, criterion, optimizer, use_cuda, epoch, n_classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    n_correct, n_total = 0, 0
    interval = c.INTERVAL
    
    model.train()

    end = time.time()
    # pbar = tqdm(enumerate(train_loader)) 
    for batch_idx, (data) in enumerate(train_loader):
        inputs, targets = data  
        targets = targets.view(-1)
        current_sample = inputs.size(0)
        
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        _, output = model(inputs) 

        n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
        n_total += current_sample
        train_acc_temp = 100. * n_correct / n_total
        train_acc.update(train_acc_temp, inputs.size(0))
        
        loss = criterion(output, targets)
        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % interval == 0:
            print(
                    'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.avg:.5f}\t'
                    'Acc {train_acc.avg:.5f}'.format(
                        epoch, batch_idx * len(inputs), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), 
                        batch_time=batch_time, loss=losses, train_acc=train_acc))
    return losses.avg
                     
def validate(val_loader, model, criterion, use_cuda, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    val_acc = AverageMeter()
    n_correct, n_total = 0, 0
    
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data) in enumerate(val_loader):
            inputs, targets = data
            current_sample = inputs.size(0)  
            
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda() 
            _, output = model(inputs)

            n_correct += (torch.max(output, 1)[1].long().view(targets.size()) == targets).sum().item()
            n_total += current_sample
            val_acc_temp = 100. * n_correct / n_total
            val_acc.update(val_acc_temp, inputs.size(0))
            
            loss = criterion(output, targets)
            losses.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        
        print('  * Validation: '
                  'Loss {loss.avg:.5f}\t'
                  'Acc {val_acc.avg:.5f}'.format(
                  loss=losses, val_acc=val_acc)) 
    return losses.avg

def main():
    use_cuda = c.USE_CUDA 
    val_ratio = c.VAL_RATIO 
    embedding_size = c.EMBEDDING_SIZE
    start = c.START 
    n_epochs = c.N_EPOCHS 
    end = start + n_epochs 
    lr = c.LR 
    wd = c.WD 
    optimizer_type = c.OPTIMIZER  
    batch_size = c.BATCH_SIZE 
    valid_batch_size = c.VAL_BATCH_SIZE 
    use_shuffle = c.USE_SHUFFLE 
    
    train_dataset, valid_dataset, n_classes = load_dataset(val_ratio)
    print('\nNumber of classes (speakers):\n{}\n'.format(n_classes))
    
    log_dir = c.MODEL_SAVED_DIR 
    #suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    #basename = 'm'
    #basename = '_'.join([basename, suffix])
    #log_dir = log_dir + basename
    print(log_dir)   

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    model = custom_model(embedding_size=embedding_size, num_classes=n_classes)

    if c.RESUME:
        checkpoint = torch.load(log_dir + '/checkpoint_' + str(c.RESUME_CHECK_POINT) + '.pth')
        state_dict = checkpoint['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    
    if use_cuda:
        print ("CUDA_VISIBLE_DEVICES:",c.CUDA_VISIBLE_DEVICES)
        os.environ['CUDA_VISIBLE_DEVICES'] = c.CUDA_VISIBLE_DEVICES
        model = nn.DataParallel(model).cuda()
        #model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(optimizer_type, model, lr, wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=c.FACTOR, patience=c.LR_PATIENCE, min_lr=c.MIN_LR, verbose=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=use_shuffle)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                       batch_size=valid_batch_size,
                                                       shuffle=False,
                                                       collate_fn=collate_fn_feat_padded)
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=c.ES_PATIENCE, verbose=True)

    #summary = SummaryWriter()
    for epoch in range(start, end):
        train_loss = train(train_loader, model, criterion, optimizer, use_cuda, epoch, n_classes)
        valid_loss = validate(valid_loader, model, criterion, use_cuda, epoch)
        scheduler.step(valid_loss, epoch)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            print('Saved dir: ', log_dir)
            break

        #if epoch % 1 == 0:
        #    summary.add_scalar('loss/train_loss', train_loss, epoch)
        #    summary.add_scalar('loss/valid_loss', valid_loss, epoch)
        #    summary.add_scalar('learning_rate', lr, epoch)

        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   '{}/checkpoint_{}.pth'.format(log_dir, epoch))

    minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
    print('Lowest validation loss at epoch %d' % minposs)
    
    visualize_the_losses(avg_train_losses, avg_valid_losses)

if __name__ == '__main__':
    main()
