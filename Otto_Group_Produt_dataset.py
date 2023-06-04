import torch
from torch import nn, optim, from_numpy
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/otto/train.csv')
train_set, test_set = train_test_split(df, test_size = 0.2, random_state=42)
train_set.head()

categories = train_set['target'].unique()

class OttoDataset(Dataset):
    def __init__(self, df):
        self.df = df
        class_map = {'Class_1' : 0, 'Class_2' : 1, 'Class_3' : 2, 'Class_4' : 3, 'Class_5' : 4,
                     'Class_6' : 5, 'Class_7' : 6, 'Class_8' : 7, 'Class_9' : 8}
        self.df['target'] = self.df['target'].map(class_map).astype(int)
        self.features = torch.Tensor(self.df.iloc[:,1:-1].values)
        self.targets = torch.Tensor(self.df.iloc[:,-1].values).view(-1,1) 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y 
    
train_dataset = OttoDataset(train_set)
test_dataset = OttoDataset(test_set)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

class OttoClassifier(nn.Module):
    def __init__(self, node_num, active_func):
        super(OttoClassifier, self).__init__()
        self.node_num = node_num
        self.active_func = active_func
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(93,self.node_num[0][0]))
        for node in node_num:
            self.layers.append(nn.Linear(node[0],node[1]))
        self.layers.append(nn.Linear(self.node_num[-1][-1],9))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = self.active_func(layer(out))
        logits = self.layers[-1](out)
        probas = self.softmax(logits)
        return logits, probas
    
def train(model, train_dataloader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        print(epoch)
        for input, label in train_dataloader:
            logits, probas = model(input)
            label = label.squeeze(-1).long()
            _, pred_labels = torch.max(probas, 1)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, optimizer, criterion

test_node_num = [(128,256),(256,512),(512,1024),(1024,512),(512,256),(256,128),(128,64)]

model = OttoClassifier(test_node_num, nn.ReLU())
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
model, optimizer, criterion = train(model, train_loader, optimizer, criterion, 100)

acc = 0
total = 0
for x, y in test_loader:
    logits, probas = model(x)
    for pro, y in zip(probas.tolist(), y.tolist()):
        if (pro.index(max(pro))==int(y[0])):
            acc += 1
        total += 1
        
print("acc :", acc/total)