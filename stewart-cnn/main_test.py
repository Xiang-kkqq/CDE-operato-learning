import torch.nn as nn 
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math
from CNN1DNetwork import CNN1D
# from ResNet import CNN1D
from tqdm import tqdm
import datetime
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

# 导入数据
print("downloading cluster data...")

date = str(datetime.datetime.now())
date = date[:date.rfind(":")].replace("-", "") \
            .replace(":", "") \
            .replace(" ", "_")

# data = loadmat('./data_stewartv10/T8.mat')
# dataset = data['T8']
# # data = loadmat('./data_try/T1.mat')
# # dataset = data['T1']

# log_dir="./logs/"+ date
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# tensorboard_writer = SummaryWriter(log_dir=log_dir)

for i in range(1, 9):
    data = loadmat(f'./data/data_stewartv13/T{i}.mat')
    dataset = data[f'T{i}']
    log_dir=f"./logs/{date}_T{i}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_writer = SummaryWriter(log_dir=log_dir)
    #6-6
    # dataset= (dataset)*1000

    # 按行（axis = 0）去除重复数据，按照行排序好
    dataset = np.unique(dataset, axis=0)

    # 划分数据集并缩放数据
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, 0:6], dataset[:, 6:13], test_size=0.2, random_state=20)
    # X_train, X_test, y_train, y_test = train_test_split(dataset[:, 0:6], dataset[:, 6:62], test_size=0.2, random_state=20)

    # scaler = StandardScaler()
    # input_train_scaled = scaler.fit_transform(X_train)
    # input_test_scaled =scaler.fit_transform(X_test)
    # 不缩放数据
    input_train_scaled = X_train
    output_train_scaled =y_train
    input_test_scaled =X_test
    output_test_scaled =y_test

    # 将numpy数组转成torch的张量
    input_train_tensor = torch.from_numpy(input_train_scaled).float()
    output_train_tensor = torch.from_numpy(output_train_scaled).float()
    input_test_tensor = torch.from_numpy(input_test_scaled).float()
    output_test_tensor = torch.from_numpy(output_test_scaled).float()

    dataset = TensorDataset(input_train_tensor,output_train_tensor)
    trainDataLoader = DataLoader(dataset, batch_size=64,shuffle=True)

    dataset = TensorDataset(input_test_tensor,output_test_tensor)
    testDataLoader = DataLoader(dataset, batch_size=64,shuffle=True)
    # 初始化模型、优化器和损失函数
    net = CNN1D()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    loss_func = RMSELoss()
    # loss_func = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1,verbose=False)

    # GPU训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    loss_func = loss_func.to(device)

    # 画图准备
    x = []                  # 横坐标
    train_loss_list = []    # train损失值
    test_loss_list = []     # test损失值

    max_epoch = 1000
    net.train()
    for epoch in tqdm(range(max_epoch)):
        for input_batch, output_batch in tqdm(trainDataLoader):
            net.train()
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)
            
            
            prediction = net(input_batch.unsqueeze(1))
            loss = loss_func(prediction.squeeze(1), output_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            # x.append(epoch)
            # print(torch.sqrt(loss))
            # train_loss_list.append(loss.data.cpu().numpy())

            # 计算测试集损失值
            net.eval()  # 切换到评估模式
            with torch.no_grad():  # 不计算梯度
                for test_input_batch,test_output_batch in tqdm(testDataLoader):
                    test_input_batch = input_test_tensor.to(device)
                    test_output_batch = output_test_tensor.to(device)
                    test_prediction = net(test_input_batch.unsqueeze(1))
                    test_loss = loss_func(test_prediction.squeeze(1), test_output_batch)
                    # test_loss_list.append(test_loss.data.cpu().numpy())
        

        tensorboard_writer.add_scalar("loss/train_train_loss", loss, epoch)
        tensorboard_writer.add_scalar("loss/test_loss", test_loss, epoch)

# loss_angle = loss_func(prediction[:, 3:6], output_train_tensor[:, 3:6])
# print('角度预测RMSE:')
# print(torch.sqrt(loss_angle))
# loss_position = loss_func(prediction[:, 0:3], output_train_tensor[:, 0:3])
# print('位移预测RMSE:')
# print(torch.sqrt(loss_position))
# print("********************************************************************")
# print("测试集的角度预测RMSE:")
# print((loss_func(test_prediction[:, 3:6], output_test_tensor[:, 3:6])).data.cpu())
# print("测试集的位移预测RMSE:")
# print((loss_func(test_prediction[:, 0:3], output_test_tensor[:, 0:3])).data.cpu())

    input_test_tensor = input_test_tensor.to(device)
    net.eval()
    # 测试网络模型
    test_prediction = net(input_test_tensor.unsqueeze(1))

    test_prediction = test_prediction.data.cpu().numpy()

    # # 反归一化
    # test_prediction = scaler.inverse_transform(test_prediction)
    # output_test_tensor = scaler.inverse_transform(output_test_tensor)

    # 把预测数据与label数据化成torch的tensor形式
    test_prediction = torch.from_numpy(test_prediction).float()
    # output_test_tensor = torch.from_numpy(output_test_tensor).float()
    output_test_tensor = torch.Tensor(output_test_tensor).float()

    # print("********************************************************************")
    print("测试集的角度预测RMSE:")
    # print(torch.sqrt(loss_func(test_prediction[:, 3:6], output_test_tensor[:, 3:6])).data.cpu() / math.pi * 180)
    # print(torch.sqrt(loss_func(test_prediction[:, 3:6], output_test_tensor[:, 3:6])).data.cpu())
    print((loss_func(test_prediction[:, 3:7], output_test_tensor[:, 3:7])).data.cpu())
    print("测试集的位移预测RMSE:")
    # print(torch.sqrt(loss_func(test_prediction[:, 0:3], output_test_tensor[:, 0:3])).data.cpu())
    print((loss_func(test_prediction[:,0:3], output_test_tensor[:,0:3])).data.cpu())

    6-56
    # print("测试集的位移预测RMSE1:")
    # print(torch.sqrt(loss_func(test_prediction[:, 0:3], output_test_tensor[:, 0:3])).data.cpu())
    # print("测试集的位移预测RMSE2:")
    # print(torch.sqrt(loss_func(test_prediction[:, 7:10], output_test_tensor[:, 7:10])).data.cpu())
    # print("测试集的位移预测RMSE3:")
    # print(torch.sqrt(loss_func(test_prediction[:, 14:17], output_test_tensor[:, 14:17])).data.cpu())
    # print("测试集的位移预测RMSE4:")
    # print(torch.sqrt(loss_func(test_prediction[:, 21:24], output_test_tensor[:, 21:24])).data.cpu())
    # print("测试集的位移预测RMSE5:")
    # print(torch.sqrt(loss_func(test_prediction[:, 28:31], output_test_tensor[:, 28:31])).data.cpu())
    # print("测试集的位移预测RMSE6:")
    # print(torch.sqrt(loss_func(test_prediction[:, 35:38], output_test_tensor[:, 35:38])).data.cpu())
    # print("测试集的位移预测RMSE7:")
    # print(torch.sqrt(loss_func(test_prediction[:, 42:45], output_test_tensor[:, 42:45])).data.cpu())
    # print("测试集的位移预测RMSE8:")
    # print(torch.sqrt(loss_func(test_prediction[:, 49:52], output_test_tensor[:, 49:52])).data.cpu())

    # print("测试集的角度预测RMSE1:")
    # print(torch.sqrt(loss_func(test_prediction[:, 3:7], output_test_tensor[:, 3:7])).data.cpu())
    # print("测试集的角度预测RMSE2:")
    # print(torch.sqrt(loss_func(test_prediction[:, 10:14], output_test_tensor[:, 10:14])).data.cpu())
    # print("测试集的角度预测RMSE3:")
    # print(torch.sqrt(loss_func(test_prediction[:, 17:21], output_test_tensor[:, 17:21])).data.cpu())
    # print("测试集的角度预测RMSE4:")
    # print(torch.sqrt(loss_func(test_prediction[:, 24:28], output_test_tensor[:, 24:28])).data.cpu())
    # print("测试集的角度预测RMSE5:")
    # print(torch.sqrt(loss_func(test_prediction[:, 31:35], output_test_tensor[:, 31:35])).data.cpu())
    # print("测试集的角度预测RMSE6:")
    # print(torch.sqrt(loss_func(test_prediction[:, 38:42], output_test_tensor[:, 38:42])).data.cpu())
    # print("测试集的角度预测RMSE7:")
    # print(torch.sqrt(loss_func(test_prediction[:, 45:49], output_test_tensor[:, 45:49])).data.cpu())
    # print("测试集的角度预测RMSE8:")
    # print(torch.sqrt(loss_func(test_prediction[:, 52:56], output_test_tensor[:, 52:56])).data.cpu())

    torch.save(net, f'net_T{i}_sin_0513_v13.pkl')
# torch.save(net, 'net_T8_sin_0408.pkl')
# torch.save(net, 'net_test_0408.pkl')