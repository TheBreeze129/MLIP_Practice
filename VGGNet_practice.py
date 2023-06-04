#import

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 80
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

import numpy as np
imgs = np.array([img.numpy() for img, _ in train_dataset])
print(f'shape: {imgs.shape}')

# 평균 계산
mean_r = np.mean(imgs, axis=(2, 3))[:, 0].mean()
mean_g = np.mean(imgs, axis=(2, 3))[:, 1].mean()
mean_b = np.mean(imgs, axis=(2, 3))[:, 2].mean()
print(mean_r, mean_g, mean_b)

# 표준편차 계산
std_r = np.std(imgs, axis=(2, 3))[:, 0].std()
std_g = np.std(imgs, axis=(2, 3))[:, 1].std()
std_b = np.std(imgs, axis=(2, 3))[:, 2].std()
print(std_r, std_g, std_b)

# new_transform define
new_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((mean_r, mean_g, mean_b),(std_r,std_g,std_b))
])

# cifar-100 download.
cf100_train_dataset = torchvision.datasets.CIFAR100(root='../../data/', train=True, transform=transform, download = True)
cf100_test_dataset = torchvision.datasets.CIFAR100(root='../../data/', train=False, transform=transforms.ToTensor())
import numpy as np
img100s = np.array([img.numpy() for img, _ in cf100_train_dataset])
print(f'shape: {img100s.shape}')
# 평균 계산
mean_r100 = np.mean(img100s, axis=(2, 3))[:, 0].mean()
mean_g100 = np.mean(img100s, axis=(2, 3))[:, 1].mean()
mean_b100 = np.mean(img100s, axis=(2, 3))[:, 2].mean()
print(mean_r100, mean_g100, mean_b100)
# 표준편차 계산
std_r100 = np.std(img100s, axis=(2, 3))[:, 0].std()
std_g100 = np.std(img100s, axis=(2, 3))[:, 1].std()
std_b100 = np.std(img100s, axis=(2, 3))[:, 2].std()
print(std_r100, std_g100, std_b100)

# dataset redefine.
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((mean_r100, mean_g100, mean_b100),(std_r100,std_g100,std_b100))
])
cf100_train_dataset = torchvision.datasets.CIFAR100(root='../../data/', train=True, transform=transform, download = True)
train_dataset = torchvision.datasets.CIFAR10(root='../../data/', train=True, transform=transform, download = True)
test_dataset = torchvision.datasets.CIFAR10(root='../../data/', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.maxpool = nn.MaxPool2d(4)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels = 16, kernel_size = 3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size = 3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1568, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

model = VGG().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(num_epochs):
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(i+1) % 100 == 0:
                print("Epoch [{}/{}], step [{}/{}] Loss: {:4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        if (epoch+1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the odel on the test images: {} %'.format(100 * correct / total))

cf100_train_loader = torch.utils.data.DataLoader(dataset= train_dataset,
                                                 batch_size = 100,
                                                 shuffle = True)

cf100_test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size=100,
                                                shuffle=False)

class VGG_3(nn.Module):
    def __init__(self, pool_stride=2):
        super(VGG_3, self).__init__()
        self.pool_stride = pool_stride
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 2, stride = self.pool_stride)   
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size = 3, padding = 1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 2, stride = self.pool_stride)
        )
        self.fc1 = nn.Linear(256*8*8,4096)
        self.fc2 = nn.Linear(4096,100)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = torch.flatten(out,1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = VGG_3().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Code

for epoch in range(1, 10):
    train(epoch)
    test()