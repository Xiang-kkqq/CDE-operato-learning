# CNN1D.py
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv6 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(32)
        self.conv7 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm1d(16)
        self.conv8 = nn.Conv1d(16, 4, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm1d(4)
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.fc1 = nn.Linear(12, 1)  
        # self.fc1 = nn.Linear(384, 128)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.softplus(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.softplus(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.softplus(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.softplus(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.softplus(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.softplus(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.softplus(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.softplus(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

