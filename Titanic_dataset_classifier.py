# custom dataloader
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TitanicDataset(Dataset):
    def __init__(self, df):
        df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # 특이한거 찾아서 바꾸기.
        df['Title'] = df['Title'].replace(["Countess", "Lady", "Mlle", "Mme", "Ms"], "FemaleRare")
        df['Title'] = df['Title'].replace(['Dr'],'Rare')
        df['Title'] = df['Title'].replace(["Capt", "Col", "Don", "Jonkheer", "Major", "Rev", "Sir"], "MaleRare")
        title_mapping = {"FemaleRare": 1, "Mrs": 2, "Miss": 3, "Master": 4, "Rare": 5, "MaleRare": 6, "Mr": 7}
        df['Title'] = df['Title'].map(title_mapping)
        df['Title'] = df['Title'].fillna(0)
        df = df.drop(['Name','PassengerId','Cabin','Ticket'], axis = 1)
        sex_mapping = {'male' : 0, 'female' : 1}
        df['Sex'] = df['Sex'].map(sex_mapping).astype(int)
        # 나이 채우기, 성별, 등급별로
        age_guess = np.zeros((2,3))
        for i in range(2):
            for j in range(3):
                guess_df = df[(df["Sex"] == i) & (df["Pclass"] == j + 1)]["Age"].dropna()
                age_guess[i,j] = round(guess_df.median())
        for i in range(2):
            for j in range(3):
                df.loc[(df["Age"].isnull()) & (df["Sex"] == i) & (df["Pclass"] == j + 1), "Age"] = age_guess[i, j]
        
        df["Age"] = df["Age"].astype(int) # 연령을 정수형으로 변환
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = 0
        df.loc[df['FamilySize']==1, 'IsAlone'] = 1
        freq_port = df["Embarked"].dropna().mode()[0]
        df["Embarked"] = df["Embarked"].fillna(freq_port)
        embarked_map = {'C' : 0, 'S' : 1, 'Q' : 2}
        df['Embarked'] = df["Embarked"].map(embarked_map).astype(int)
        df['Fare'] = df["Fare"].fillna(df["Fare"].dropna().median())

        self.features = torch.FloatTensor(df.iloc[:,1:].values)
        self.targets = torch.Tensor(df.iloc[:,0].values).view(-1,1)       

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y
    
    def feature_size(self):
        return self.features.shape[-1]
    
df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/titanic/train.csv')
df2 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/titanic/test.csv')
df3 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/data/titanic/gender_submission.csv')
df2['Survived'] = df3['Survived']
df2 = df[['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]
train_set = TitanicDataset(df)
test_set = TitanicDataset(df2)
train_loader = DataLoader(train_set, batch_size=32,shuffle=True)
test_loader = DataLoader(test_set, batch_size=32,shuffle=True)

import torch
import torch.nn.functional as F
from torch import nn, optim

class Model(nn.Module):
    def __init__(self, node_num, active_func):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.node_num = node_num
        self.active_func = active_func
        self.layer_num = len(node_num)
        self.layers.append(nn.Linear(10,self.node_num[0][0]))
        for layer in node_num:
            self.layers.append(nn.Linear(layer[0],layer[1]))
        self.layers.append(nn.Linear(self.node_num[-1][-1],1))

    def forward(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = self.active_func(layer(out))
        out = self.layers[-1](out)
        out = nn.Sigmoid()(out)
        return out
    
def train(model, train_dataloader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        for input, label in train_dataloader:
            y_pred = model(input)
            loss = criterion(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, optimizer, criterion

active_funcs = [nn.ReLU(), nn.ReLU6(), nn.ELU(), nn.SELU(), nn.PReLU(), nn.LeakyReLU(), nn.Threshold(0.1, 0.5),
                nn.Hardtanh(), nn.Sigmoid(), nn.Tanh()]

test_node_num = [(8,16),(16,32),(32,64),(64,128),(128,256),(256,128),(128,64)]

model = Model(test_node_num, nn.ReLU())
criterion = nn.BCELoss(reduction = 'mean')
optimizer = optim.Adam(model.parameters(), lr = 0.01)

model, optimizer, criterion = train(model, train_loader, optimizer, criterion, 100)

# 정확도 체크.
total = 0
acc = 0
for x, y in test_loader:
    y_pred = model(x)
    for a, b in zip(y_pred.tolist(),y.tolist()):
        if (round(a[0]) ==int(b[0])):
            acc += 1
        total += 1
print(acc/total)