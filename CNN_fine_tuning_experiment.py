import torch
import torch.nn as nn
from torch import nn, optim, from_numpy, cuda
import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
print(device)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(1.0,))
])

# mnist download
train_data = datasets.MNIST(root = './data/02/', train=True, download=True, transform = train_transform)
test_data = datasets.MNIST(root = './data/02/', train=False, download=True, transform = transforms.ToTensor())

batch_size = 1024
train_loader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset=test_data, batch_size = batch_size, shuffle = True)

class MNIST_CNN(nn.Module):
    def __init__(self, feature_shape, active_func, convs_kernel_size_list, convs_out_list, out_dim_size):
        super(MNIST_CNN, self).__init__()
        self.feature_shape = feature_shape
        self.convs = nn.ModuleList()
        self.active_func = active_func
        self.output_active_func = nn.Softmax(dim=-1)
        for i in range(len(convs_out_list)):
            if i==0:
                self.convs.append(nn.Conv2d(feature_shape[0], convs_out_list[i], kernel_size = convs_kernel_size_list[i]))
            else:
                self.convs.append(nn.Conv2d(convs_out_list[i-1],convs_out_list[i], kernel_size = convs_kernel_size_list[i]))
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(self._compute_out_channels(), out_dim_size)

    def _compute_out_channels(self):
        dim_0 = self.convs[-1].out_channels
        dim_1 = self.feature_shape[1]
        dim_2 = self.feature_shape[2]
        for conv in self.convs:
            dim_1 -= (conv.kernel_size[0] -1) 
            dim_2 -= (conv.kernel_size[0] -1)
            dim_1 = dim_1//2
            dim_2 = dim_2//2
        return (dim_0*dim_1*dim_2)//self.feature_shape[0]

    def forward(self, x):
        out = x
        for layer in self.convs:
            out = self.active_func(self.mp(layer(out)))
        out = torch.flatten(out, 1)
        logit = self.fc(out)
        proba = self.output_active_func(logit)
        return logit, proba
            
def train(model, train_loader, optimizer, criterion, num_epochs):
    pppr_loss = 1000
    ppr_loss = 1000
    pr_loss = 1000
    for epoch in range(num_epochs):
        for batch_ids, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            logits, proba = model(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            if(loss > ppr_loss and loss > pr_loss and loss > pppr_loss):
                return model, optimizer, criterion
            pppr_loss = ppr_loss
            ppr_loss = pr_loss
            pr_loss = loss.item()
            loss.backward()
            optimizer.step()
    return model, optimizer, criterion         

model = MNIST_CNN(train_data[0][0].shape, nn.ReLU(), convs_kernel_size_list, convs_out_list, 10)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(reduction = 'mean')

model, optimizer, criterion = train(model, train_loader, optimizer, criterion, 5)

def accuracy_check(model, optimizer, criterion, test_loader):
    total = 0
    correct = 0
    with torch.no_grad():
        for features, target in test_loader:
            features = features.to(device)
            target = target.to(device)

            _, probas = model(features)
            _, predicted = torch.max(probas.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the model on the test images: {:.2f}%'.format(accuracy))
    return accuracy

accuracy_check(model, optimizer, criterion, test_loader)

#파라미터를 변경하며 해보겠습니다.
NUM_EPOCHS = 5
convs_out_list = [5, 10, 20]
convs_kernel_size_list = [2, 3]
layer_nums = [2,3]
active_funcs = [nn.ReLU(), nn.ELU(), nn.Tanh()]
lrs = [1e-01, 1e-02, 1e-03]
optimizers = [optim.Adam, optim.SGD]
loss_funcs = [nn.CrossEntropyLoss, nn.NLLLoss]

import random

hyper_params = []
acc_history_hp = []

# 하이퍼파라미터들을 반복문으로 돌며 모델 학습과 평가를 수행
for optimizer in optimizers:
    for loss_func in loss_funcs:
        for lr in lrs:
            for active_func in active_funcs:
                for layer_num in layer_nums:
                    kernel_size = [convs_kernel_size_list[random.randint(0, 1)] for _ in range(layer_num)]
                    
                    

                    # 모델 생성
                    model = MNIST_CNN(train_data[0][0].shape, active_func, kernel_size, convs_out_list[:layer_num], 10)

                    # 옵티마이저, 손실 함수 설정
                    if optimizer == optim.Adam:
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                    elif optimizer == optim.SGD:
                        optimizer = optim.SGD(model.parameters(), lr=lr)

                    criterion = loss_func()

                    # 모델 학습
                    model, optimizer, criterion = train(model, train_loader, optimizer, criterion, 20)

                    # 모델 평가
                    acc = accuracy_check(model, optimizer, criterion, test_loader)
                    acc_history_hp.append(acc)
                    hyper_param = [loss_func, optimizer, lr, active_func, layer_num, kernel_size]
                    hyper_params.append(hyper_param)

best_acc = max(acc_history_hp)
best_hp = hyper_params[acc_history_hp.index(best_acc)]
best_hp

import pandas as pd
ac_his = np.array(acc_history_hp)
hps = np.array(hyper_params)
df = pd.DataFrame(hps, columns=['loss_func', 'optimizer', 'lr', 'active_func', 'layer_num', 'kernel_size'])
df['accuracy'] = ac_his
map_optim = {df['optimizer'].unique().tolist()[0] : 'Adam', df['optimizer'].unique().tolist()[1] : 'SGD'}
df['optimizer'] = df['optimizer'].map(map_optim)
map_loss = {df['loss_func'].unique().tolist()[0] : 'CELoss', df['loss_func'].unique().tolist()[1] : 'NLLLoss'}
df['loss_func'] = df['loss_func'].map(map_loss)

# a. layer 개수별 분석
for i in df['layer_num'].unique().tolist():
    print(f"layer_num == {i}'s mean acc : {df[df['layer_num']==i]['accuracy'].mean()}")
# layer 2개인게 더 높습니다.

# b. 활성 함수별 분석
for i in df['active_func'].unique():
    print(f"active_func == {i}'s mean acc : {df[df['active_func']==i]['accuracy'].mean()}")
# ReLU가 가장 높습니다.

# c. lr별 분석
for i in df['lr'].unique():
    print(f"lr == {i}'s mean acc : {df[df['lr']==i]['accuracy'].mean()}")

# c. optimizer별 분석
for i in df['optimizer'].unique():
    print(f"optimizer == {i}'s mean acc : {df[df['optimizer']==i]['accuracy'].mean()}")

# d. loss 함수별 분석
for i in df['loss_func'].unique():
    print(f"loss_funcr == {i}'s mean acc : {df[df['loss_func']==i]['accuracy'].mean()}")

# 위를 조합해 최적인 경우의 수.
df.head(1)
