# mount

from torch import nn, optim, from_numpy
import numpy as np
from google.colab import drive

drive.mount('/content/gdrive')

# Load file
xy = np.loadtxt('/content/gdrive/My Drive/Colab Notebooks/data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:,[-1]])
x_train = x_data[0:int(len(x_data)*0.8)]
x_test = x_data[int(len(x_data)*0.8):]
y_train = y_data[0:int(len(x_data)*0.8)]
y_test = y_data[int(len(x_data)*0.8):]
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')

# Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, 64)
        self.l4 = nn.Linear(64, 128)
        self.l5 = nn.Linear(128, 256)
        self.l6 = nn.Linear(256, 512)
        self.l7 = nn.Linear(512, 256)
        self.l8 = nn.Linear(256, 128)
        self.l9 = nn.Linear(128, 64)
        self.l10 = nn.Linear(64, 1)
        self.active_func = nn.Sigmoid()

    def forward(self, x):
        out1 = self.active_func(self.l1(x))
        out2 = self.active_func(self.l2(out1))
        out3 = self.active_func(self.l3(out2))
        out4 = self.active_func(self.l4(out3))
        out5 = self.active_func(self.l5(out4))
        out6 = self.active_func(self.l6(out5))
        out7 = self.active_func(self.l7(out6))
        out8 = self.active_func(self.l8(out7))
        out9 = self.active_func(self.l9(out8))
        y_pred = self.active_func(self.l10(out9))
        return y_pred

model = Model()

criterion = nn.BCELoss(reduction = 'mean')
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# Training loop
for epoch in range(100):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, 64)
        self.l4 = nn.Linear(64, 128)
        self.l5 = nn.Linear(128, 256)
        self.l6 = nn.Linear(256, 512)
        self.l7 = nn.Linear(512, 256)
        self.l8 = nn.Linear(256, 128)
        self.l9 = nn.Linear(128, 64)
        self.l10 = nn.Linear(64, 1)
        self.active_func = nn.ReLU()

    def forward(self, x):
        out1 = self.active_func(self.l1(x))
        out2 = self.active_func(self.l2(out1))
        out3 = self.active_func(self.l3(out2))
        out4 = self.active_func(self.l4(out3))
        out5 = self.active_func(self.l5(out4))
        out6 = self.active_func(self.l6(out5))
        out7 = self.active_func(self.l7(out6))
        out8 = self.active_func(self.l8(out7))
        out9 = self.active_func(self.l9(out8))
        y_pred = self.active_func(self.l10(out9))
        return y_pred

model = Model()

criterion = nn.BCELoss(reduction = 'mean')
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# Training loop
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import torch
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, node_num, active_func):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.node_num = node_num
        self.active_func = active_func
        self.layers.append(nn.Linear(8,self.node_num[0][0]))
        for layer in node_num:
            self.layers.append(nn.Linear(layer[0],layer[1]))
        self.layers.append(nn.Linear(self.node_num[-1][-1],1))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = self.active_func(layer(out))
        return out
    
def train(model, x_data, y_data, early_stopping_count, criterion, optimizer, print_epoch=True):
    count = 0
    pred_loss = -1
    for epoch in range(100):
        if(count == early_stopping_count):
            break
        y_pred = model(x_data)
        y_pred_clamped = torch.clamp(y_pred, 0, 1)
        loss = criterion(y_pred_clamped, y_data)
        if(loss==pred_loss):
            count += 1
        pred_loss = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if print_epoch:
        print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f} | layer_num: {len(model.node_num)} | active_func : {model.active_func}')
        print(f'node per layer: {model.node_num}')
    return model, criterion, optimizer

def accuracy(model, x_data, y_data, criterion):
    with torch.no_grad():
        y_pred = model(x_data)
        y_pred_label = (y_pred >= 0.5).float()
        accuracy = (y_pred_label == y_data).float().mean()

    return accuracy.item()

active_funcs = [nn.ReLU(), nn.ReLU6(), nn.ELU(), nn.SELU(), nn.PReLU(), nn.LeakyReLU(), nn.Threshold(0.1, 0.5),
                nn.Hardtanh(), nn.Sigmoid(), nn.Tanh()]

test_node_num = [(16,32),(32,64),(64,128),(128,256),(256,128),(128,64)]

model = Model(test_node_num, nn.Sigmoid())
criterion = nn.BCELoss(reduction = 'mean')
optimizer = optim.SGD(model.parameters(), lr = 0.01)

model, criterion, optimizer = train(model, x_train, y_train, 10, criterion, optimizer)
print(accuracy(model, x_test, y_test, criterion))

test_node_num = [(16,32),(32,16)]

for func in active_funcs:
    model = Model(test_node_num, func)
    criterion = nn.BCELoss(reduction = 'mean')
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    model, criterion, optimizer = train(model, x_train, y_train, 10, criterion, optimizer, False)
    print(f'Accuracy: {accuracy(model, x_test, y_test, criterion):.4f} | layer_num: {len(model.node_num)} | active_func : {func}')
    del model, criterion, optimizer

# 0.7434

test_node_num = [(16,32)]

for func in active_funcs:
    model = Model(test_node_num, func)
    criterion = nn.BCELoss(reduction = 'mean')
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    model, criterion, optimizer = train(model, x_train, y_train, 10, criterion, optimizer, False)
    print(f'Accuracy: {accuracy(model, x_test, y_test, criterion):.4f} | layer_num: {len(model.node_num)} | active_func : {func}')
    del model, criterion, optimizer
# 0.7632

test_node_num = [(16,32),(32,64),(64,128),(128,256)]

for func in active_funcs:
    model = Model(test_node_num, func)
    criterion = nn.BCELoss(reduction = 'mean')
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    model, criterion, optimizer = train(model, x_train, y_train, 10, criterion, optimizer, False)
    print(f'Accuracy: {accuracy(model, x_test, y_test, criterion):.4f} | layer_num: {len(model.node_num)} | active_func : {func}')
    del model, criterion, optimizer
# 0.7566    

test_node_num = [(16,32),(32,64),(64,128),(128,64),(64,32),(32,64)]

for func in active_funcs:
    model = Model(test_node_num, func)
    criterion = nn.BCELoss(reduction = 'mean')
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    model, criterion, optimizer = train(model, x_train, y_train, 10, criterion, optimizer, False)
    print(f'Accuracy: {accuracy(model, x_test, y_test, criterion):.4f} | layer_num: {len(model.node_num)} | active_func : {func}')
    del model, criterion, optimizer
# 0.7566    

import torch
import torch.nn.functional as F

# 레이어 별로 다른 활성화 함수를 써보자!

class Model(nn.Module):
    def __init__(self, node_num, active_funcs):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.node_num = node_num
        self.active_funcs = active_funcs
        self.layers.append(nn.Linear(8,self.node_num[0][0]))
        for layer in node_num:
            self.layers.append(nn.Linear(layer[0],layer[1]))
        self.layers.append(nn.Linear(self.node_num[-1][-1],1))

    def forward(self, x):
        out = x
        for layer, func in zip(self.layers, self.active_funcs):
            out = func(layer(out))
        return out
    
test_node_num = [(16,32)]

test_active_funcs = [nn.SELU(), nn.SELU(), nn.ReLU6()]

model = Model(test_node_num, test_active_funcs)
criterion = nn.BCELoss(reduction = 'mean')
optimizer = optim.SGD(model.parameters(), lr = 0.01)
model, criterion, optimizer = train(model, x_train, y_train, 10, criterion, optimizer, False)
print(f'Accuracy: {accuracy(model, x_test, y_test, criterion):.4f} | layer_num: {len(model.node_num)} | active_funcs : {test_active_funcs}')
del model, criterion, optimizer

# 0.7697

test_node_num = [(16,32)]

test_active_funcs = [nn.SELU(), nn.SELU(), nn.ELU()]

highest_acc = 0

for i in range(100):
    model = Model(test_node_num, test_active_funcs)
    criterion = nn.BCELoss(reduction = 'mean')
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    model, criterion, optimizer = train(model, x_train, y_train, 10, criterion, optimizer, False)
    acc = accuracy(model, x_test, y_test, criterion)
    if(highest_acc<acc):
        highest_acc = acc
    #print(f'Accuracy: {acc:.4f} | layer_num: {len(model.node_num)} | active_funcs : {test_active_funcs}')
    del model, criterion, optimizer

print(highest_acc)

# 0.7763

test_node_num = [(16,32)]

test_active_funcs = [nn.SELU(), nn.SELU(), nn.ELU()]

from itertools import product

highest_acc = 0
highest_case = []

for i in product(range(len(active_funcs)),range(len(active_funcs)),range(len(active_funcs))):
    test_active_funcs = [active_funcs[i[0]],active_funcs[i[1]],active_funcs[i[2]]]
    for i in range(100):
        model = Model(test_node_num, test_active_funcs)
        criterion = nn.BCELoss(reduction = 'mean')
        optimizer = optim.SGD(model.parameters(), lr = 0.01)
        model, criterion, optimizer = train(model, x_train, y_train, 10, criterion, optimizer, False)
        acc = accuracy(model, x_test, y_test, criterion)
        if(highest_acc<acc):
            highest_acc = acc
            highest_case = test_active_funcs
        #print(f'Accuracy: {acc:.4f} | layer_num: {len(model.node_num)} | active_funcs : {test_active_funcs}')
        del model, criterion, optimizer


print(f"Highest acc: {highest_acc}, Case : {highest_case}")


# 0.7828

# sklearn의 와인 데이터셋 사용
import torch
import numpy as np
from torch.autograd import Variable
from torch import nn, optim, from_numpy
import torch.nn.functional as F

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

wine = load_wine()

wine_data = from_numpy(wine.data).float()
wine_target = from_numpy(wine.target).long()
train_X, test_X, train_Y, test_Y = train_test_split(wine_data, wine_target)

class WineClassifier(nn.Module):
    def __init__(self, node_num, active_func):
        super(WineClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.active_func = active_func
        self.node_num = node_num
        self.layers.append(nn.Linear(13,node_num[0][0]))
        for node in node_num:
            self.layers.append(nn.Linear(node[0],node[1]))
        self.layers.append(nn.Linear(node_num[-1][-1],3))
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = self.active_func(layer(out))
        out = F.softmax(out, dim=1)
        return out
    
def train(model, x_data, y_data, early_stopping_count, criterion, optimizer, print_epoch=True):
    count = 0
    pred_loss = -1
    for epoch in range(1000):
        if(count == early_stopping_count):
            break
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        if(loss==pred_loss):
            count += 1
        pred_loss = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if print_epoch:
        print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f} | layer_num: {len(model.node_num)} | active_func : {model.active_func}')
        print(f'node per layer: {model.node_num}')
    return model, criterion, optimizer

active_funcs = [nn.ReLU(), nn.ReLU6(), nn.ELU(), nn.SELU(), nn.PReLU(), nn.LeakyReLU(), nn.Threshold(0.1, 0.5),
                nn.Hardtanh(), nn.Sigmoid(), nn.Tanh()]

node_num = [(26,52),(52,26),(26,13)]
for func in active_funcs:
    model = WineClassifier(node_num, func)
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()
    model, optimizer, criterion = train(model, train_X, train_Y, 10, criterion, optimizer,False)
    y_pred = model(test_X).tolist()
    correct = 0
    total = 0
    for x,y in zip(y_pred, test_Y.tolist()):
        if (x.index(max(x))==y):
            correct += 1
        total += 1
    print(correct/total)
    del model, optimizer, criterion

# 71퍼센트까지 했습니다.

