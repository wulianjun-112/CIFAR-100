import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision.transforms import transforms
import torchvision
import torch.multiprocessing as mp
import torch.utils.data.distributed
import os
import argparse
from model import ResNet34

from utils import progress_bar

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# 创建一个解析器
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
# 添加lr参数，默认=1
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--warm_up',
                    '-w',
                    action='store_true',
                    default=False,
                    help='warm_up by learning rate=0.01')
# 添加参数,是否加载checkpoint，继续训练
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--local_rank", type=int)
# 解析参数
args = parser.parse_args()
rank = args.local_rank
print("rank：", rank)
torch.cuda.set_device(rank)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
train_sampler = torch.utils.data.distributed.DistributedSampler(
    trainset, num_replicas=2, rank=args.local_rank)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=False,
                                          num_workers=4,
                                          sampler=train_sampler)
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
net = net.to(rank)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True
torch.distributed.init_process_group(backend="nccl")
net = torch.nn.parallel.DistributedDataParallel(net,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank)
# device_ids will include all GPU devices by default

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
                                                       patience=5,
                                                       verbose=False,
                                                       threshold=0.0001,
                                                       threshold_mode='rel',
                                                       cooldown=2,
                                                       min_lr=0,
                                                       eps=1e-08)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
#                                                        T_max=200)  # 待调
# train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
#                                                  milestones=[1, 3, 5],
#                                                  gamma=0.2)


def Warmup_LR(optimizer, epoch, use_warmup=False):
    """Warmup"""
    factor = epoch / 10
    if (factor <= 1):
        lr = 0.1 * (torch.distributed.get_world_size()**factor)
        if use_warmup:
            if epoch < 3:
                factor = epoch / 2
                lr = args.lr * (torch.distributed.get_world_size()**factor)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# Training
def train(rank, epoch):
    if (rank == 0):
        print('\nEpoch: %d' % epoch)
    Warmup_LR(optimizer, epoch, use_warmup=args.warm_up)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(rank), targets.to(rank)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (rank == 0):
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
            inputs, targets = inputs.to(rank), targets.to(rank)
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
    current_lr = optimizer.param_groups[0]['lr']
    trainloader.sampler.set_epoch(epoch)
    train(rank, epoch)
    if (rank == 0):
        test(epoch)
    scheduler.step(best_acc)
    if (current_lr != optimizer.param_groups[0]['lr'] and epoch > 10):
        print('==> Resuming from checkpoint again..')
        assert os.path.isdir(
            'checkpoint-100'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint-100/ckpt-100.pth')
        net.load_state_dict(checkpoint['net'])
        optimizer.param_groups[0]['lr'] = current_lr * 0.2
