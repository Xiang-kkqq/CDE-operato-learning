import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, l1_reg=0.08, l2_reg=0.1):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(32, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(512,256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(256,128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv1d(128,64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(64)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv1d(64,32, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm1d(32)
        self.pool7 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv8 = nn.Conv1d(32,8, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm1d(8)
        self.leakyrelu = nn.LeakyReLU()
        # self.fc1 = nn.Linear(48, 48)
        self.fc1 = nn.Linear(48, 7)

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leakyrelu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.leakyrelu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.leakyrelu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.leakyrelu(x)
        x = x.view(x.size(0), -1)
        
        # Apply L1 regularization to each layer's output
        # l1_loss = 0
        # for param in self.parameters():
        #     l1_loss += torch.sum(torch.abs(param))

        # l2_loss = 0
        # for param in self.parameters():
        #     l2_loss += torch.sum(param.pow(2))
        
        
        x = self.fc1(x)
        
        # Add L1 loss to the total loss
        # l1_loss *= self.l1_reg
        # # Multiply L2 loss by 0.5 to match the standard form
        # l2_loss *= 0.5 * self.l2_reg

        # return x + l1_loss + l2_loss
        return x




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CNN1D(nn.Module):
#     def __init__(self, l1_reg=0.08, l2_reg=0.2):
#         super(CNN1D, self).__init__()
#         self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.conv4 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm1d(128)
#         self.conv5 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm1d(64)
#         self.conv6 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm1d(32)
#         self.conv7 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)
#         self.bn7 = nn.BatchNorm1d(16)
#         self.conv8 = nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1)
#         self.bn8 = nn.BatchNorm1d(8)
#         self.leakyrelu = nn.LeakyReLU()
#         self.fc1 = nn.Linear(48, 24) 
#         self.fc2 = nn.Linear(24, 6)  # Output layer

#         self.l1_reg = l1_reg
#         self.l2_reg = l2_reg

#     def forward(self, x): 
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.leakyrelu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.leakyrelu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.leakyrelu(x)
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.leakyrelu(x)
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.leakyrelu(x)
#         x = self.conv6(x)
#         x = self.bn6(x)
#         x = self.leakyrelu(x)
#         x = self.conv7(x)
#         x = self.bn7(x)
#         x = self.leakyrelu(x)
#         x = self.conv8(x)
#         x = self.bn8(x)
#         x = self.leakyrelu(x)
#         x = x.view(x.size(0), -1)
        
#         # Apply L1 regularization to each layer's output
#         l1_loss = 0
#         for param in self.parameters():
#             l1_loss += torch.sum(torch.abs(param))

#         l2_loss = 0
#         for param in self.parameters():
#             l2_loss += torch.sum(param.pow(2))
        
#         x = self.fc1(x)
#         x = self.leakyrelu(x)
#         x = self.fc2(x)
        
#         # Add L1 and L2 losses to the total loss
#         l1_loss *= self.l1_reg
#         l2_loss *= 0.5 * self.l2_reg

#         return x + l1_loss + l2_loss




# import torch
# import torch.nn as nn

# class CNN1D(nn.Module):
#     def __init__(self):
#         super(CNN1D, self).__init__()
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.conv4 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm1d(64)
#         self.conv5 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm1d(32)
#         # self.conv6 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
#         # self.bn6 = nn.BatchNorm1d(64)
#         # self.conv7 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
#         # self.bn7 = nn.BatchNorm1d(64)
#         # self.conv8 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
#         # self.bn8 = nn.BatchNorm1d(32)
#         self.leakyrelu = nn.LeakyReLU()
#         # self.sigmoid = nn.Sigmoid()
#         self.fc1 = nn.Linear(192,6)
        
        
#     def forward(self, x): 
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.leakyrelu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.leakyrelu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.leakyrelu(x)
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.leakyrelu(x)
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.leakyrelu(x)
#         # x = self.conv6(x)
#         # x = self.bn6(x)
#         # x = self.leakyrelu(x)
#         # x = self.conv7(x)
#         # x = self.bn7(x)
#         # x = self.leakyrelu(x)
#         # x = self.conv8(x)
#         # x = self.bn8(x)
#         x = self.leakyrelu(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x


# import torch
# import torch.nn as nn

# class CNN1D(nn.Module):
#     def __init__(self, dropout_rate=0.2):
#         super(CNN1D, self).__init__()
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.dropout1 = nn.Dropout(p=dropout_rate)
        
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.dropout2 = nn.Dropout(p=dropout_rate)
        
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.dropout3 = nn.Dropout(p=dropout_rate)
        
#         self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm1d(256)
#         self.dropout4 = nn.Dropout(p=dropout_rate)
        
#         self.conv5 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm1d(512)
#         self.dropout5 = nn.Dropout(p=dropout_rate)
        
#         self.conv6 = nn.Conv1d(512, 128, kernel_size=3, stride=1, padding=1)
#         self.bn6 = nn.BatchNorm1d(128)
#         self.dropout6 = nn.Dropout(p=dropout_rate)
        
#         self.conv7 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
#         self.bn7 = nn.BatchNorm1d(64)
#         self.dropout7 = nn.Dropout(p=dropout_rate)
        
#         self.conv8 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
#         self.bn8 = nn.BatchNorm1d(32)
#         self.dropout8 = nn.Dropout(p=dropout_rate)
        
#         self.leakyrelu = nn.LeakyReLU()
#         self.fc1 = nn.Linear(192, 6)
        
        
#     def forward(self, x): 
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.dropout1(x)
#         x = self.leakyrelu(x)
        
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.dropout2(x)
#         x = self.leakyrelu(x)
        
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.dropout3(x)
#         x = self.leakyrelu(x)
        
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.dropout4(x)
#         x = self.leakyrelu(x)
        
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.dropout5(x)
#         x = self.leakyrelu(x)
        
#         x = self.conv6(x)
#         x = self.bn6(x)
#         x = self.dropout6(x)
#         x = self.leakyrelu(x)
        
#         x = self.conv7(x)
#         x = self.bn7(x)
#         x = self.dropout7(x)
#         x = self.leakyrelu(x)
        
#         x = self.conv8(x)
#         x = self.bn8(x)
#         x = self.dropout8(x)
#         x = self.leakyrelu(x)
        
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x


