#import

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class LayerNormalization(nn.Module):
    def __init__(self, features, epsilon=1e-8):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        ln = (x - mean) / (std + self.epsilon)
        ln = self.gamma * ln + self.beta
        return ln
    
class InstanceNormalization(nn.Module):
    def __init__(self, num_features, epsilon=1e-8):
        super(InstanceNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_features, 1, 1))
        self.epsilon = epsilon
    
    def forward(self, x):
        mean = x.mean((2, 3), keepdim=True)
        std = torch.sqrt(x.var((2, 3), keepdim=True) + self.epsilon)
        inorm = (x - mean) / std
        inorm = self.gamma * inorm + self.beta
        return inorm
    
class GroupNormalization(nn.Module):
    def __init__(self, num_channels, num_groups, epsilon=1e-8):
        super(GroupNormalization, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    
    def forward(self, x):
        x = x.view(-1, self.num_groups, self.num_channels // self.num_groups, *x.size()[2:])
        mean = x.mean((2, 3, 4), keepdim=True)
        std = torch.sqrt(x.var((2, 3, 4), keepdim=True) + self.epsilon)
        gnorm = (x - mean) / std
        gnorm = gnorm.view(-1, self.num_channels, *x.size()[2:])
        gnorm = self.gamma * gnorm + self.beta
        return gnorm
    
class ConditionalNormalization(nn.Module):
    def __init__(self, num_features, num_conditions, epsilon=1e-8):
        super(ConditionalNormalization, self).__init__()
        self.num_features = num_features
        self.num_conditions = num_conditions
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.condition_scale = nn.Linear(num_conditions, num_features)
        self.condition_shift = nn.Linear(num_conditions, num_features)
    
    def forward(self, x, condition):
        mean = x.mean((2, 3), keepdim=True)
        std = torch.sqrt(x.var((2, 3), keepdim=True) + self.epsilon)
        c_scale = self.condition_scale(condition).unsqueeze(2).unsqueeze(3)
        c_shift = self.condition_shift(condition).unsqueeze(2).unsqueeze(3)
        c_norm = (x - mean) / std
        c_norm = c_scale * c_norm + c_shift
        c_norm = self.gamma * c_norm + self.beta
        return c_norm
    
