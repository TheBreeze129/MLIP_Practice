import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2**10

transform__ = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform__)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform__)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.5, 0.5, 0.5] + np.array([0.5, 0.5, 0.5]))
    image = image.clip(0, 1)
    return image

class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
        self.branch_pool_2d = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    
    def out_channel_dim(self):
        return self.branch1x1.out_channels + self.branch5x5_2.out_channels + self.branch3x3dbl_3.out_channels*2 # *2 for pool_2d

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5_1 = self.branch5x5_1(x)
        branch5x5_2 = self.branch5x5_2(branch5x5_1)

        branch3x3dbl_1 = self.branch3x3dbl_1(x)
        branch3x3dbl_2 = self.branch3x3dbl_2(branch3x3dbl_1)
        branch3x3dbl_3 = self.branch3x3dbl_3(branch3x3dbl_2)

        branch_pool_2d = self.branch_pool_2d(x)
        branch_pool = self.branch_pool(branch_pool_2d)

        outputs = [branch1x1, branch5x5_2, branch3x3dbl_3, branch_pool]
        outputs = torch.cat(outputs, 1)
        return outputs
    
class Net(nn.Module):
    def __init__(self, feature_dims, out_dim, inceptions_in_channels):       
        super(Net, self).__init__()
        self.feature_dims = feature_dims
        self.num_inceptions = len(inceptions_in_channels)
        self.convs = nn.ModuleList()
        self.inceptions = nn.ModuleList()

        inception_out_channels = 88 # 이게 가장 좋다네요.
        
        for i in range(len(inceptions_in_channels)):
            if i == 0:
                self.convs.append(nn.Conv2d(feature_dims[0], inceptions_in_channels[i], kernel_size=3))
            else:
                self.convs.append(nn.Conv2d(inception_out_channels, inceptions_in_channels[i], kernel_size=3))
            self.inceptions.append(Inception(inceptions_in_channels[i]))
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(self._compute_inceptions_out_channels(), out_dim)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def _compute_inceptions_out_channels(self):
        dim_0 = 88  
        dim_1 = self.feature_dims[1]
        dim_2 = self.feature_dims[2]
        for conv in self.convs:
            dim_1 -= (conv.kernel_size[0]-1)    
            dim_2 -= (conv.kernel_size[0]-1)
            dim_1 = dim_1//2
            dim_2 = dim_2//2
        return int(dim_0 * dim_1 * dim_2)

    def forward(self, x):
        for i in range(self.num_inceptions):
            x = self.activation(self.mp(self.convs[i](x)))
            x = self.inceptions[i](x)
        
        x = torch.flatten(x, 1)
        logit = self.fc(x)
        proba = self.softmax(logit)
        
        return logit, proba
    
incep_in_channels = [5, 10, 20]
lr = 1e-02
NUM_EPOCHS = 5

criterion = nn.CrossEntropyLoss()
num_layers_list = [1, 2, 3, 4,]

best_loss = []
for num_layers in num_layers_list:
    model = Net(train_data[0][0].shape, 10, incep_in_channels[:num_layers])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(NUM_EPOCHS):   
        batch_loss_history = [] 
        for features, targets in train_loader:
            logits, probas = model(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad()        
            loss.backward()
            optimizer.step()
            batch_loss_history.append(loss.cpu().detach().numpy())
        losses.append(sum(batch_loss_history)/len(batch_loss_history))
    best_loss.append(min(losses))

import matplotlib.pyplot as plt

plt.bar(num_layers_list, best_loss)
plt.xticks(num_layers_list)

