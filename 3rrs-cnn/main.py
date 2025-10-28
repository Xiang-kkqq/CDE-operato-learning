import torch.nn as nn 
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pyswarms as ps
from sklearn.model_selection import train_test_split
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
import torch.nn.functional as F

from CNN1DNetwork import CNN1D


# 导入数据
print("downloading cluster data...")
data = loadmat('./data/data_0110/cluster2_kmeans.mat')
dataset = data['cluster2']

dataset = np.unique(dataset, axis=0)

# 划分数据集并缩放数据
X_train, X_test, y_train, y_test = train_test_split(dataset[:, 0:3], dataset[:, 8:9], test_size=0.2, random_state=20)

# 不缩放数据【不归一化】
input_train_scaled = X_train
output_train_scaled = y_train
input_test_scaled = X_test
output_test_scaled = y_test


# # 将numpy数组转成torch的张量
input_train_tensor = torch.from_numpy(input_train_scaled).float()
output_train_tensor = torch.from_numpy(output_train_scaled).float()
input_test_tensor = torch.from_numpy(input_test_scaled).float()
output_test_tensor = torch.from_numpy(output_test_scaled).float()

dataset = TensorDataset(input_train_tensor,output_train_tensor)
DataLoader = DataLoader(dataset, batch_size=32,shuffle=True)

# 初始化模型、优化器和损失函数
net = CNN1D()


optimizer = torch.optim.Adam(net.parameters(), lr=0.001,)
loss_func = nn.MSELoss()

# GPU训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
loss_func = loss_func.to(device)

# 画图准备
x = []                  # 横坐标
train_loss_list = []    # train损失值
val_loss_list = []      # val损失值
# train_acc_list = []     # train准确率


# GPU训练
input_train_tensor = input_train_tensor.to(device)
output_train_tensor = output_train_tensor.to(device)

max_epoch = 50

for epoch in range(max_epoch):
    for input_batch, output_batch in DataLoader:
        input_batch, output_batch = input_batch.to(device), output_batch.to(device)
        optimizer.zero_grad()
        prediction = net(input_batch.unsqueeze(1))
        # prediction = net(input_batch)
        # loss = loss_func(prediction.squeeze(1), output_batch)
        RMSEloss = torch.sqrt(loss_func(prediction, output_batch))
        RMSEloss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        x.append(epoch)
        train_loss_list.append(RMSEloss.data.cpu().numpy())
        # val_loss_list.append(torch.sqrt(val_loss))

        # 绘制训练过程
        # 允许动态画图
        plt.ion()
        plt.cla()   # 清除坐标轴
        try:
            train_loss_lines.remove(train_loss_lines[0])    # 移除上一步曲线
        except Exception:
            pass

        train_loss_lines = plt.plot(x, train_loss_list, 'r', lw=1)  # lw为曲线宽度
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["train_loss"])
        plt.pause(0.1)
        
    if epoch == max_epoch - 1:
        # 取出预测数据与label数据并化成numpy形式
        prediction = net(input_train_tensor.unsqueeze(1))    # 前向传播
        # prediction = net(input_train_tensor)
        prediction = prediction.data.cpu().numpy()
        output_train_tensor = output_train_tensor.data.cpu().numpy()
        
        # 把预测数据与label数据化成torch的tenser形式
        prediction = torch.from_numpy(prediction).float()
        output_train_tensor = torch.from_numpy(output_train_tensor).float()

        loss_position = loss_func(prediction[0:,], output_train_tensor[0:,])
        print('位移预测RMSE:')
        print(torch.sqrt(loss_position))
        # print('例子:')
        # print(prediction[0, 3:6], output_train_tensor[0, 3:6])

plt.ioff()
plt.show()
plt.savefig('train_loss')
plt.close()

input_test_tensor = input_test_tensor.to(device)

# 测试网络模型
test_prediction = net(input_test_tensor.unsqueeze(1))
# test_prediction = net(input_test_tensor)

test_prediction = test_prediction.data.cpu().numpy()

test_prediction = torch.from_numpy(test_prediction).float()
# output_test_tensor = torch.from_numpy(output_test_tensor).float()
output_test_tensor = torch.Tensor(output_test_tensor).float()

print("********************************************************************")
print("测试集的角度预测RMSE:")
print(torch.sqrt(loss_func(test_prediction[0], output_test_tensor[0])).data.cpu() / math.pi * 180)
print("测试集的位移预测RMSE:")
print(torch.sqrt(loss_func(test_prediction[0:,], output_test_tensor[0:,])).data.cpu())

# 保存网络模型结构以及参数
torch.save(net, 'net_cluster1_1212_6.pkl')

# 取出模型
net_record = torch.load('net.pkl')
