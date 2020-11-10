import argparse
import os
import random
import shutil
import warnings
import torch
import torch.nn as nn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import utils
from progressbar import Bar
import modellist
import torchvision.datasets as dsets
import copy

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='dir', help='path to dataset')
parser.add_argument('-m', '--modelname', metavar='ARCH', default='resnet18', help='model architecture: ')
parser.add_argument('-a', '--attention', metavar='attention', default='', help='attention')
parser.add_argument('--numclasses', default=1000, type=int, metavar='C', help='num classes')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batchsize', default=32, type=int,
                    metavar='N', help='batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--hgpath', default='', type=str, metavar='PATH',
                    help='history graph path')

history_dict = {'train':{'acc':[],'loss':[]},'val':{'acc':[],'loss':[]}}
best_acc1 = 0

def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    global history_dict
    global best_acc1
    history_path = ''
    
    device = torch.device('cuda')
    if device is not None:
        print("Use GPU: {} for training".format(device))
    
    if args.attention == 'se':
        attention_module='se_layer'
    elif args.attention == 'cbam':
        attention_module='cbam_layer'
    else:
        attention_module = None 
        
    model = modellist.Modellist(args.modelname, args.numclasses, attention_module)
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        model.to(device)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            history_dict = checkpoint['history_dict']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    best_acc_wts = copy.deepcopy(model.state_dict())
    

    # Data loading code
    ########################################################################
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])


    train_dataset = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                            train=True, # True를 지정하면 훈련 데이터로 다운로드
                            transform=transform, # 텐서로 변환
                            download=True)

    test_dataset = dsets.MNIST(root='MNIST_data/', 
                            train=False, 
                            transform=transform, 
                            download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = args.batchsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = args.batchsize, shuffle=False)

    #######################################################

    if args.evaluate:
        val(val_loader, model, criterion, device, best_acc_wts)
        return

    for epoch in range(args.start_epoch, args.epochs):

        utils.adjust_learning_rate(optimizer, epoch, args.start_epoch)
        before_best_acc1=best_acc1

        # train for one epoch
        print('epochs :',epoch+1,'/',args.epochs)
        train(train_loader, model, criterion, optimizer, device)
        # evaluate on validation set
        best_acc_wts = val(val_loader, model, criterion, device, best_acc_wts)
        model.load_state_dict(best_acc_wts)
        is_best = before_best_acc1 < best_acc1

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.modelname,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'history_dict' : history_dict,
            }, is_best, history_path)
    if args.hgpath:        
        utils.train_graph(args.epochs, history_dict, args.hgpath)
    
def train(train_loader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(Bar(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.data.cpu().numpy()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = train_loss /len(train_loader)
    epoch_acc = correct / total

    history_dict['train']['loss'].append(epoch_loss)
    history_dict['train']['acc'].append(epoch_acc)

    print('train | Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))


def val(val_loader, model, criterion, device,best_acc_wts):
    global best_acc1
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(Bar(val_loader)):
            inputs,targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())     
            test_loss += loss.data.cpu().numpy()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()    
    epoch_loss = test_loss / len(val_loader)
    epoch_acc = correct / total

    history_dict['val']['loss'].append(epoch_loss)
    history_dict['val']['acc'].append(epoch_acc)

    print('val | Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))
    if best_acc1 < epoch_acc:
        best_acc1 = epoch_acc
        best_acc_wts = copy.deepcopy(model.state_dict())
    return best_acc_wts







if __name__ == '__main__':
    main()
