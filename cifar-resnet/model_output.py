import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import pickle
from tqdm import tqdm
import pickle


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

def model_output_f(logits, labels):
    bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
    logits_correct = logits[bindex, labels]

    cloned_logits = logits.clone()
    cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

    margins = logits_correct - cloned_logits.logsumexp(dim=-1)
    return -margins



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model-ckpt', type=str, help='learning rate')
    parser.add_argument('--epoch', type=str, help='rho')
    parser.add_argument('--exp-id', type=str, help='rho')
    args = parser.parse_args()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=8)
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test)
    
    random_indices = random.sample(range(len(testset)), int(len(testset)*0.3))

    noisy_test_set=torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    for i in random_indices:
        current_label = noisy_test_set.targets[i]
        noisy_label = noisy_test_set.targets[i]
        while noisy_label == current_label:
            noisy_label = random.randint(0, 10 - 1)
        noisy_test_set.targets[i] = noisy_label
    noisy_test_loader = torch.utils.data.DataLoader(
        noisy_test_set, batch_size=128, shuffle=False, num_workers=8)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=False, num_workers=8)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss(reduction='none')
    train_loss_matrix = []
    test_loss_matrix = []
    train_prob_matrix = []
    test_prob_matrix = []
    train_margin_matrix = []
    test_margin_matrix = []
    train_logits_matrix = []
    test_logits_matrix = []
    for dirpath, dirnames, filenames in tqdm(os.walk(args.model_ckpt)):
        for file in filenames:
            # Full path to the file
            file_path = os.path.join(dirpath, file)
            if not file_path.endswith(f'{args.epoch}.pth'): continue

            net = construct_rn9()
            net.load_state_dict(torch.load(file_path))
            net.to(device)
            net.eval()
            loss_list=[]
            prob_list=[]
            margin_list=[]
            logits_list=[]
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    margin = model_output_f(outputs, targets)
                    bindex = torch.arange(outputs.shape[0]).to(device=outputs.device, non_blocking=False)
                    prob = F.softmax(outputs, dim=-1)[bindex, targets]
                    logits = outputs[bindex, targets]
                    loss_list.append(loss)
                    prob_list.append(prob)
                    margin_list.append(margin)
                    logits_list.append(logits)
            loss_list = torch.cat(loss_list)
            prob_list = torch.cat(prob_list)
            margin_list = torch.cat(margin_list)
            logits_list = torch.cat(logits_list)
            train_loss_matrix.append(loss_list)
            train_prob_matrix.append(prob_list)
            train_margin_matrix.append(margin_list)
            train_logits_matrix.append(logits_list)
            
            loss_list=[]
            prob_list=[]
            margin_list=[]
            logits_list=[]
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    margin = model_output_f(outputs, targets)
                    bindex = torch.arange(outputs.shape[0]).to(device=outputs.device, non_blocking=False)
                    prob = F.softmax(outputs, dim=-1)[bindex, targets]
                    logits = outputs[bindex, targets]
                    loss_list.append(loss)
                    prob_list.append(prob)
                    margin_list.append(margin)
                    logits_list.append(logits)
            loss_list = torch.cat(loss_list)
            prob_list = torch.cat(prob_list)
            margin_list = torch.cat(margin_list)
            logits_list = torch.cat(logits_list)
            test_loss_matrix.append(loss_list)
            test_prob_matrix.append(prob_list)
            test_margin_matrix.append(margin_list)
            test_logits_matrix.append(logits_list)
        
    train_loss_matrix = torch.stack(train_loss_matrix).cpu()
    test_loss_matrix = torch.stack(test_loss_matrix).cpu()
    train_prob_matrix = torch.stack(train_prob_matrix).cpu()
    test_prob_matrix = torch.stack(test_prob_matrix).cpu()
    train_margin_matrix = torch.stack(train_margin_matrix).cpu()
    test_margin_matrix = torch.stack(test_margin_matrix).cpu()
    train_logits_matrix = torch.stack(train_logits_matrix).cpu()
    test_logits_matrix = torch.stack(test_logits_matrix).cpu()


    with open(f'{args.model_ckpt}/losses-{args.exp_id}-on-train.pkl', 'wb') as out:
        pickle.dump(train_loss_matrix, out)
    
    with open(f'{args.model_ckpt}/losses-{args.exp_id}-on-test.pkl', 'wb') as out:
        pickle.dump(test_loss_matrix, out)

    with open(f'{args.model_ckpt}/prob-{args.exp_id}-on-train.pkl', 'wb') as out:
        pickle.dump(train_prob_matrix, out)

    with open(f'{args.model_ckpt}/prob-{args.exp_id}-on-test.pkl', 'wb') as out:
        pickle.dump(test_prob_matrix, out)

    with open(f'{args.model_ckpt}/margin-{args.exp_id}-on-train.pkl', 'wb') as out:
        pickle.dump(train_margin_matrix, out)

    with open(f'{args.model_ckpt}/margin-{args.exp_id}-on-test.pkl', 'wb') as out:
        pickle.dump(test_margin_matrix, out)

    with open(f'{args.model_ckpt}/logits-{args.exp_id}-on-train.pkl', 'wb') as out:
        pickle.dump(train_logits_matrix, out)

    with open(f'{args.model_ckpt}/logits-{args.exp_id}-on-test.pkl', 'wb') as out:
        pickle.dump(test_logits_matrix, out)