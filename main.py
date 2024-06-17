'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from datetime import datetime
import json
import sys

torch.manual_seed(42)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint') # resume from the previous checkpoint

parser.add_argument('--train_batch', default=128, type=int, help='train batch_size')
parser.add_argument('--test_batch', default=100, type=int, help='test batch_size')
parser.add_argument('--epoch', default=200, type=int, help='total training epochs')
# VGG19, ResNet18, PreActResNet18, GoogLeNet, DenseNet121, ResNeXt29_2x64d, MobileNet, MobileNetV2
# DPN92, ShuffleNetG2, SENet18, ShuffleNetV2, EfficientNetB0, RegNetX_200MF, SimpleDLA
parser.add_argument('--net_type', default='SimpleDLA', type=str, help='model type')
# SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Adamax
parser.add_argument('--optim_type', default='SGD', type=str, help='optimizer type')
# CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, StepLR, ExponentialLR, LambdaLR, MultiStepLR, CyclicLR
parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', type=str, help='learning rate scheduler type')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global_best_acc = 0  # best test accuracy
local_best_acc = 0
local_best_list = []
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
start_time = datetime.now().strftime('%y%m%d_%H%M%S')

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.train_batch, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

networks = {
    'VGG19': lambda: VGG('VGG19'),
    'ResNet18': ResNet18,
    'PreActResNet18': PreActResNet18,
    'GoogLeNet': GoogLeNet,
    'DenseNet121': DenseNet121,
    'ResNeXt29_2x64d': ResNeXt29_2x64d,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'DPN92': DPN92,
    'ShuffleNetG2': ShuffleNetG2,
    'SENet18': SENet18,
    'ShuffleNetV2': lambda: ShuffleNetV2(1),
    'EfficientNetB0': EfficientNetB0,
    'RegNetX_200MF': RegNetX_200MF,
    'SimpleDLA': SimpleDLA
}
if args.net_type in networks:
    net = networks[args.net_type]()
else:
    raise ValueError(f'Unknown network type: {args.net_type}')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

'''
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    all_files = os.listdir('./checkpoint/')
    filtered_files = [file for file in all_files if file.startswith(args.net_type)]
    filtered_file = ''
    if len(filtered_files) < 1:
        print(f'No {args.net_type} model exists')
        sys.exit(1)
    else:
        filtered_file = str(os.path.splitext(filtered_files[0])[0])
        checkpoint = torch.load('./checkpoint/' + filtered_file)
        net.load_state_dict(checkpoint['net'])
        global_best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
'''

# read json
json_dict = {}
json_full = './checkpoint/model_info/' + args.net_type + '.json'
if os.path.exists(json_full):
    with open(json_full, 'r') as f:
        json_dict = json.load(f)
        global_best_acc = json_dict["best"][0]
print(f'previous best acc: {global_best_acc}')


criterion = nn.CrossEntropyLoss()

# ~6.9 AM
'''
optimizers = {
    'SGD': lambda params: optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4),
    'Adam': lambda params: optim.Adam(params, lr=args.lr, weight_decay=5e-4),
    'AdamW': lambda params: optim.AdamW(params, lr=args.lr, weight_decay=5e-4),
    'RMSprop': lambda params: optim.RMSprop(params, lr=args.lr, weight_decay=5e-4),
    'Adagrad': lambda params: optim.Adagrad(params, lr=args.lr, weight_decay=5e-4),
    'Adadelta': lambda params: optim.Adadelta(params, lr=args.lr, weight_decay=5e-4),
    'Adamax': lambda params: optim.Adamax(params, lr=args.lr, weight_decay=5e-4),
}
'''
# 6.9 PM ~
optimizers = {
    'SGD': lambda params: optim.SGD(params, lr=args.lr, momentum=0.95, weight_decay=1e-4),
    'Adam': lambda params: optim.Adam(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-5),
    'AdamW': lambda params: optim.AdamW(params, lr=args.lr, weight_decay=1e-4),
    'RMSprop': lambda params: optim.RMSprop(params, lr=args.lr, alpha=0.9, weight_decay=1e-4),
    'Adagrad': lambda params: optim.Adagrad(params, lr=args.lr, weight_decay=1e-4),
    'Adadelta': lambda params: optim.Adadelta(params, lr=args.lr, weight_decay=1e-4),
    'Adamax': lambda params: optim.Adamax(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5),
}
if args.optim_type in optimizers:
    optimizer = optimizers[args.optim_type](net.parameters())
else:
    raise ValueError(f'Unknown optimizer type: {args.optim_type}')

schedulers = {
    'CosineAnnealingLR': lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200),
    'ReduceLROnPlateau': lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10),
    'OneCycleLR': lambda opt: torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=1000),
    'StepLR': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.1),
    'ExponentialLR': lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95),
    'LambdaLR': lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.95 ** epoch),
    'MultiStepLR': lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 80], gamma=0.1),
    'CyclicLR': lambda opt: torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.001, max_lr=0.01, step_size_up=500),
}
if args.lr_scheduler in schedulers:
    scheduler = schedulers[args.lr_scheduler](optimizer)
else:
    raise ValueError(f'Unknown lr_scheduler type: {args.lr_scheduler}')


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def set_local_best_info(acc, now, ckpt_name, epoch):
    global local_best_list

    model_info = [acc]
    model_params = {
        "start_time": start_time,
        "current_time": now,
        "lr_rate": args.lr,
        "train_batch": args.train_batch,
        "test_batch": args.test_batch,
        "optim_type": args.optim_type,
        "lr_scheduler": args.lr_scheduler,
        "current_epoch(+1)": epoch+1,
        "total_epoch_setting": args.epoch,
    }
    model_info.append(model_params)
    model_info.append(ckpt_name)

    local_best_list = model_info

def save_model_info(acc, now, ckpt_name, epoch):
    global global_best_acc

    # json - {"best" : [list1], "records" : [big_list]}
    # list1 - [acc, {dict1}, model_name]
    # big_list - [list2, list3, list4, ...]
    model_info = [acc]
    model_params = {
        "start_time": start_time,
        "current_time": now,
        "lr_rate": args.lr,
        "train_batch": args.train_batch,
        "test_batch": args.test_batch,
        "optim_type": args.optim_type,
        "lr_scheduler": args.lr_scheduler,
        "current_epoch(+1)": epoch+1,
        "total_epoch_setting": args.epoch,
    }
    model_info.append(model_params)
    model_info.append(ckpt_name)


    json_dict = {}
    json_full = './checkpoint/model_info/' + args.net_type + '.json'
    if os.path.exists(json_full):
        with open(json_full, 'r') as f:
            json_dict = json.load(f)
            global_best_acc = json_dict["best"][0]

    if acc > global_best_acc:
        print(f"global best acc update: {acc}")

    new_json = {}
    json_list = [model_info]

    if json_dict != {}:
        json_list.append(json_dict["best"])
    if "records" in json_dict:
        records_list = json_dict["records"]
        json_list.extend(records_list)
    
    json_list = sorted(json_list, key=lambda x:x[0], reverse=True)
    new_json["best"] = json_list[0]
    if len(json_list) > 1:
        new_json["records"] = json_list[1:]

    try:
        with open(json_full, 'w') as f:
            json.dump(new_json, f, indent=4)
        print("json file save success")
    except Exception as e:
        print(f"(error) json file save failed: {e}")

def save_local_best_info():
    global local_best_list

    json_dict = {}
    json_full = './checkpoint/model_info/' + args.net_type + '.json'
    if os.path.exists(json_full):
        with open(json_full, 'r') as f:
            json_dict = json.load(f)

    new_json = {}
    json_list = [local_best_list]

    if json_dict != {}:
        json_list.append(json_dict["best"])
    if "records" in json_dict:
        records_list = json_dict["records"]
        json_list.extend(records_list)
    
    json_list = sorted(json_list, key=lambda x:x[0], reverse=True)
    new_json["best"] = json_list[0]
    if len(json_list) > 1:
        new_json["records"] = json_list[1:]

    try:
        with open(json_full, 'w') as f:
            json.dump(new_json, f, indent=4)
        print("json file save success")
    except Exception as e:
        print(f"(error) json file save failed: {e}")


def save_model(ckpt_name, state):
    ckpt_full = './checkpoint/' + ckpt_name + '.pth'
    
    print('Saving..')

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    try:
        torch.save(state, ckpt_full)
        print("pth file save success")
    except Exception as e:
        print(f"(error) pth file save failed: {e}")



def test(epoch):
    global global_best_acc
    global local_best_acc

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

    if acc > local_best_acc:
        local_best_acc = acc

        model_type = args.net_type
        #acc_str = 'acc' + str(acc).replace('.', '_')
        ckpt_name = model_type + '_time' + start_time

        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        save_model(ckpt_name, state)
        now = datetime.now().strftime('%y%m%d_%H%M%S')
        set_local_best_info(acc, now, ckpt_name, epoch)

        if acc >= global_best_acc:
            global_best_acc = acc
            save_model_info(acc, now, ckpt_name, epoch)

    if (epoch == args.epoch - 1) and (acc < global_best_acc):
        save_local_best_info()


for epoch in range(start_epoch, start_epoch + args.epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
