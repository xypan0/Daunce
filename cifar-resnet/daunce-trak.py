import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import os
import argparse

import time
import random
import numpy as np
import hashlib
import io
torch.set_printoptions(precision=2, sci_mode=True)
from tqdm import tqdm

# Resnet9
class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)


def construct_rn9(num_classes=10):
    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
        return torch.nn.Sequential(
                torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, groups=groups, bias=False),
                torch.nn.BatchNorm2d(channels_out),
                torch.nn.ReLU(inplace=True)
        )
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2)
    )
    return model

class DatasetWithIdx(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the data and the index
        return self.data[idx], idx

def get_model_md5(model):
    # Retrieve the state dictionary of the model, which contains all weights and buffers
    state_dict = model.state_dict()

    # Create a BytesIO buffer to hold the serialized state_dict
    buffer = io.BytesIO()

    # Use torch.save to serialize the state_dict to the buffer
    torch.save(state_dict, buffer)

    # Reset the buffer's current position to the beginning
    buffer.seek(0)
    sz=buffer.getbuffer().nbytes
    print(f'buffer size: {sz}')
    # Read the entire content of the buffer
    model_bytes = buffer.read()

    # Compute the MD5 hash of the serialized bytes
    md5_hash = hashlib.md5(model_bytes).hexdigest()

    return md5_hash

def model_output_f(logits, labels):
    bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
    logits_correct = logits[bindex, labels]

    cloned_logits = logits.clone()
    cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

    margins = logits_correct - cloned_logits.logsumexp(dim=-1)
    return -margins


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--rho', default=1, type=float, help='rho')
    parser.add_argument('--gamma', default=1e-2, type=float, help='gamma')
    parser.add_argument('--model-ckpt', default=None, type=str, help='model checkpoint')
    parser.add_argument('--save', default=None, type=str, help='model checkpoint save dir')
    parser.add_argument('--epoch', default=1, type=int, help='number of epochs')
    parser.add_argument('--ratio', default=1.0, type=float, help='mp subset ratio')
    parser.add_argument('--bsz', default=128, type=int, help='number of epochs')
    parser.add_argument('--save_interval', default=5, type=int, help='model checkpoint saving interval')
    parser.add_argument('--pseudo_random', type=int, default=1234, help='pseudo random seed for all')
    args = parser.parse_args()

    if args.pseudo_random is not None:
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(args.pseudo_random + 1)
        np.random.seed(args.pseudo_random + 1)
        torch.manual_seed(args.pseudo_random)
        torch.cuda.manual_seed(args.pseudo_random)
        torch.cuda.manual_seed_all(args.pseudo_random)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f'set seed to {args.pseudo_random}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    trainset = DatasetWithIdx(torchvision.datasets.CIFAR10(
        root='/tmp/cifar/', train=True, download=True, transform=transform_train))
    train_subset = torch.utils.data.Subset(trainset, np.random.choice(len(trainset), int(len(trainset)*args.ratio), replace=False))
    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=args.bsz, shuffle=False, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root='/tmp/cifar/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=8)
    
    r_inputs, r_targets = testset[9]

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model Initialization
    print('==> Building model..')
    if args.model_ckpt:
        net = construct_rn9()
        base_net = construct_rn9()
        net.load_state_dict(torch.load(args.model_ckpt))
        base_net.load_state_dict(torch.load(args.model_ckpt))

    else:
        raise ValueError
    net = net.to(device)
    base_net = base_net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    train_steps = len(trainloader)*args.epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps)
    
    rand_ksi = torch.rand((len(trainset),), dtype=torch.float32).to(device)
    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        st_time = time.time()
        net.train()
        base_net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, ((inputs, targets), indexes) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_rand_ksi = rand_ksi[indexes]

            optimizer.zero_grad()
            base_net.zero_grad()
            
            outputs = net(inputs)
            model_f = model_output_f(outputs, targets)
            
            with torch.no_grad():
                base_outputs = base_net(inputs)
            base_outputs.requires_grad_(True)
            
            base_criterion = nn.CrossEntropyLoss(reduction='none')
            base_loss = base_criterion(base_outputs, targets)
            base_f = model_output_f(base_outputs, targets).detach()

            factor = args.rho * (2 * batch_rand_ksi - 1)

            grad = torch.autograd.grad(
                outputs=base_loss, 
                inputs=base_outputs,
                grad_outputs=torch.ones_like(base_loss),
                only_inputs=True,  # We're only interested in logits' gradient
                create_graph=False,  # Set to True if you need higher-order derivatives
                retain_graph=False,)
            base_loss = base_loss.detach()
            base_outputs = base_outputs.detach()
            mean_base_loss = torch.sum(base_loss) / base_loss.shape[0]

            fo = torch.sum(factor * ((outputs - base_outputs) * grad[0].detach()).sum(1)) / base_loss.shape[0]
            model_loss = criterion(outputs, targets)

            reg = torch.sum((model_f - base_f) ** 2) / base_loss.shape[0]
            loss = args.gamma * reg - fo
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            base_loss = torch.sum(base_loss).detach() / base_loss.shape[0]
            # print(f'final loss: {loss:.2f}, reg: {reg:.2f}, first order: {fo:.2f}, base loss: {mean_base_loss:.2f}, model_loss: {model_loss:.2f}, grad norm: {grad_norm}', flush=True)

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        duration=time.time()-st_time
        print('Epoch: %d | Train Loss: %.3f | Acc: %.3f%% (%d/%d) | first order term: %.3f | Time: %ds' % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total, fo, duration), flush=True)

    def test(epoch):

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

        print('Epoch: %d | Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        
        # Save checkpoint.
        if epoch % args.save_interval == 0 and args.save is not None:
            print('Saving..')
            if not os.path.isdir(args.save):
                os.mkdir(args.save)
            torch.save(net.state_dict(), f'{args.save}/ckpt-{epoch}.pth')

    for epoch in range(1, args.epoch+1):
        train(epoch)
        test(epoch)
