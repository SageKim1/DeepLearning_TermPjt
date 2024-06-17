'''Test CIFAR10 with PyTorch.'''
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--net_type', default='SimpleDLA', type=str, help='test model type')
parser.add_argument('--test_batch', default=100, type=int, help='test batch_size')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Check whether file exist or not
all_files = os.listdir('./checkpoint/')
filtered_files = [file for file in all_files if file.startswith(args.net_type)]
filtered_file = ''
if len(filtered_files) < 1:
    print(f'{args.net_type} model does not exist in the checkpoint dir.')
    sys.exit(1)
else:
    filtered_file = str(os.path.splitext(filtered_files[0])[0])
    print(f'{args.net_type} model exists in the checkpoint dir: {filtered_file}')

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Get Json
model_info = dict()
json_path = './checkpoint/model_info/' + filtered_file + '.json'
with open(json_path, 'r') as json_files:
    model_info = dict(json.load(json_files))

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
if model_info["model_type"] in networks:
    net = networks[model_info["model_type"]]()
else:
    raise ValueError(f'Unknown network type: {model_info["model_type"]}')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('==> Model loading from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint =  torch.load('./checkpoint/' + filtered_file + '.pth')
net.load_state_dict(checkpoint['net'])
global_best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()


def save_test_result(acc):
    test_info = {
        "acc": acc,
        "ckpt_name": filtered_file + '.pth',
        "test_batch": args.test_batch,
        "model_type": args.net_type,
    }

    save_full = './checkpoint/test_result/' + filtered_file + '.json'
    try:
        with open(save_full, 'w') as json_file:
            json.dump(test_info, json_file)
        print('json file saved in test_result dir.')
    except Exception as e:
        print(f"(error) json file save failed: {e}")

def test():
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

    acc = 100.*correct/total
    save_test_result(acc)

test()