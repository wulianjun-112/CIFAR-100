import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision.transforms import transforms
import torchvision
import torch.multiprocessing as mp
import os
import argparse

from model import ResNet34

from utils import progress_bar

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
# 创建一个解析器
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
# 添加lr参数，默认=1
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
# 添加参数,是否加载checkpoint，继续训练
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
# 解析参数
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
])

trainset = torchvision.datasets.CIFAR100(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=4)
cifar100_test = torchvision.datasets.CIFAR100(root='./data',
                                              train=False,
                                              download=True,
                                              transform=transform_test)
cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test,
                                                   batch_size=100,
                                                   shuffle=False,
                                                   num_workers=4)
print('==> Building model..')
net = ResNet34()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(
        'checkpoint-100'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint-100/ckpt-100.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss(reduce=False, size_average=False, reduction='sum')
optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=0.9,
                      weight_decay=5e-4)  # 权重L2正则化
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='max',
                                                       factor=0.2,
                                                       patience=10,
                                                       verbose=False,
                                                       threshold=0.0001,
                                                       threshold_mode='rel',
                                                       cooldown=5,
                                                       min_lr=0,
                                                       eps=1e-08)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
#                                                        T_max=200)  # 待调
# train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
#                                                  milestones=[1, 3, 5],
#                                                  gamma=0.2)


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

        progress_bar(
            batch_idx, len(trainloader),
            'lr= %.6f | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (optimizer.param_groups[0]['lr'], train_loss /
             (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(cifar100_test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx, len(cifar100_test_loader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (test_loss /
                 (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint-100'):
            os.mkdir('checkpoint-100')
        torch.save(state, './checkpoint-100/ckpt-100.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    scheduler.step(best_acc)
