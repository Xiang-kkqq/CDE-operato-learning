from CNN1DNetwork import CNN1D
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import autograd.numpy as np
import scipy.io
from scipy.spatial.transform import Rotation as R
from scipy.io import loadmat
from scipy.io import savemat
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import time
import random
# from jax import grad
from autograd import jacobian
from torch.autograd import grad
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def axis_angle_to_euler(sd, thd, order='XYZ'):
    # 将螺旋轴和旋转角度转换为旋转矩阵
    rot_matrix = R.from_rotvec(thd * sd).as_matrix()
    # 从旋转矩阵中提取欧拉角
    r = R.from_matrix(rot_matrix)
    euler = r.as_euler(order, degrees=True)  # degrees=True 表示输出欧拉角为度
    return euler

def TargetFunction(x, D):
 
    #定义已给出的基座及运动平台
    A = np.array([
        [1.2000, 0.8000, -0.3000, -1.2000, -0.7000, 0.6000],
        [0, 1.2000, 1.0000, 0.4000, -0.8000, -1.2000],
        [0, 0, 0, 0, 0, 0]])
    mu = 0.6
    # B = np.array([[mu * element for element in row] for row in A])
    B = mu*A
    sd = x[3:6]

    norm_sd=np.linalg.norm(x[3:6])
    sd = sd / norm_sd
    s_x, s_y, s_z = sd
    # s_x, s_y, s_z = x[3:6] / np.linalg.norm(x[3:6])
    # s_x, s_y, s_z = x[3:6]
    sth = np.sin(x[6])
    cth = np.cos(x[6])
    vth = 1 - cth
    R = np.array([
    [s_x**2 * vth + cth, s_x * s_y * vth - s_z * sth, s_x * s_z * vth + s_y * sth],
    [s_y * s_x * vth + s_z * sth, s_y**2 * vth + cth, s_y * s_z * vth - s_x * sth],
    [s_z * s_x * vth - s_y * sth, s_z * s_y * vth + s_x * sth, s_z**2 * vth + cth]
    ])
    # R = sc2rot(x[3:6],x[6])
    # BA = R*B
    BA = np.dot(R, B)

    F = np.empty((0,))

    for i in range(6):
        f_i = x[0:3] + BA[:, i] - A[:, i]
        F=np.append(F, -D[i]**2 + np.dot(f_i, f_i))
    
    # Calculate the constraint for the norm of s
    F=np.append(F, np.dot(x[3:6], x[3:6]) - 1)
    
    return F


# def TargetFunction(x, D):



def create_wrapped_TargetFunction(D):
    # x=np.array(x)
    # def wrapped_TargetFunction(px, py, pz, sx, sy, sz, th):
    def wrapped_TargetFunction(x):
        # x=np.array(x._value)
        return TargetFunction(x, D)
    return wrapped_TargetFunction



if_newton = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#读取网络模型
num_nets = 8
models =[]

for i in range(num_nets):
    # model_path = f'./log/net_T{i+1}_sin_0425.pkl'  
    model_path = f'./log/net_T{i+1}_sin_0513_v13.pkl'  
    # model_path = f'./log/net_T{i+1}_sin_0418.pkl'  
    model = torch.load(model_path, map_location=device)
    model.eval()
    models.append(model.to(device))

data = loadmat('./trajectory_test/sin_0.1/Trajectory_test_1000_0925_v1.mat')
dataset = data['T_trajectory']
data1 = loadmat('./trajectory_test/sin_0.1/Trajectory_test_1000_0925_v1_angle.mat')
dataset_angle = data1['T_trajectory_angle']


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

loss_func = RMSELoss()
loss_func = loss_func.to(device)

input_data = torch.from_numpy(dataset[:, 0:6]).float()
output_data = torch.from_numpy(dataset[:, 6:13]).float()

dataset = TensorDataset(input_data, output_data)
data_loader = DataLoader(dataset)

input_data = input_data.to(device)
output_data = output_data.to(device)

input_data_angle = torch.from_numpy(dataset_angle[:, 0:6]).float()
output_data_angle = torch.from_numpy(dataset_angle[:, 6:12]).float()

dataset_angle = TensorDataset(input_data_angle, output_data_angle)
data_loader_angle = DataLoader(dataset)

input_data_angle = input_data_angle.to(device)
output_data_angle = output_data_angle.to(device)



# 将三个轨迹的参数同时输入到八个网络中
predictions = []

T_record = []
T_record1 = []

first_iteration = True

for step in range(len(input_data)):
    
    min_distance = float('inf')

    if first_iteration:
        selected_prediction = output_data[0, :]
        first_iteration = False
        continue
    else:
        previous_prediction = selected_prediction

    # 计时
    T0 = time.time()
    
    # 记录选择的模型
    model_choose = 0
    for model_num in range(len(models)):
        with torch.no_grad():
            model_type = models[model_num]
            model_type.eval()
            input_sample = input_data[step, :].reshape(1, 6)
            prediction = model_type(input_sample.unsqueeze(1)).reshape(7)


            # 选择标准
            distance = (loss_func(prediction[0:7], previous_prediction[0:7]))

            if distance < min_distance:
                model_choose = model_num
                min_distance = distance
                selected_prediction = prediction
                # selected_prediction = selected_prediction / 1000

    if if_newton:
        print("步数：", step)
        DegreeToRad = np.pi/180
        input_sample = input_sample.cpu().numpy()[0]
        selected_prediction = selected_prediction.cpu().numpy()
        # [theta1_value, theta2_value, theta3_value] = [input_sample[0] * DegreeToRad, input_sample[1] * DegreeToRad, input_sample[2] * DegreeToRad]
        D = input_sample[0:6]

        epsilon = 1e-2
        max_iter = 100
        iter_count = 0

        # 计时
        T1 = time.time()
        
        x_last = np.array([selected_prediction[0], selected_prediction[1], selected_prediction[2], selected_prediction[3], selected_prediction[4], selected_prediction[5], selected_prediction[6]])
        # x_last = x_last / 1000
        x_cur = x_last

        # x_cur = Variable(torch.tensor(x_last, requires_grad=True))


        while iter_count < max_iter:            
            # funcs = TargetFunction(x_cur[0], x_cur[1], x_cur[2], x_cur[3], x_cur[4], x_cur[5], x_cur[6], D)
            funcs = TargetFunction(x_cur, D)

            x_norm = np.linalg.norm(funcs.astype('float'), ord=None, axis=None, keepdims=False)
            if(x_norm <= epsilon):
                break
            else:
                custom_wrapped_func = create_wrapped_TargetFunction(D)
                jacobian_custom_wrapped_func = jacobian(custom_wrapped_func)
                jacobian_func_at_x = jacobian_custom_wrapped_func(x_cur)
                # print(jacobian_func_at_x)
                inverse = np.linalg.inv(jacobian_func_at_x)
                x_last = x_cur
                x_cur = x_last - np.matmul(inverse, funcs).transpose()
                x_cur = x_cur
                iter_count += 1
        

        # 计时
        T2 = time.time()

        T_record.append((T2-T1)*1000)

        T_record1.append((T2-T0)*1000)

        print("预测一步，牛顿法的迭代时间：", (T2-T1)*1000)

        print("迭代次数：", iter_count, "\n")
        
        selected_prediction[0] = x_cur[0]
        selected_prediction[1] = x_cur[1]
        selected_prediction[2] = x_cur[2]
        selected_prediction[3] = x_cur[3]
        selected_prediction[4] = x_cur[4]
        selected_prediction[5] = x_cur[5]
        selected_prediction[6] = x_cur[6]

        sd = selected_prediction[3:6]  # 提取螺旋轴
        thd = selected_prediction[6]  # 提取旋转角度
        euler_angles = axis_angle_to_euler(sd, thd)  # 将螺旋轴和旋转角度转换为欧拉角


        selected_prediction = torch.from_numpy(selected_prediction).to(device)


    # if step >= 221 and step <= 251:
    #     print("第", step, "步:", "\n", "选择模型:", model_choose, "\n", "牛顿法后的prediction:", selected_prediction, "\n")
    # predictions.append(selected_prediction)
    predictions.append(torch.tensor([selected_prediction[0], selected_prediction[1], selected_prediction[2], euler_angles[0], euler_angles[1], euler_angles[2]]).to(device))

predictions = torch.stack(predictions)

print("牛顿法的平均时间：", sum(T_record)/len(T_record))
print("每一步预测的平均时间：", sum(T_record1)/len(T_record1))


# plt.subplot(311)
plt.plot(output_data[1::, 0].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 0].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for px')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.savefig('TrajectoryTracing_px')
plt.show()
plt.close()

# plt.subplot(312)
plt.plot(output_data[1::, 1].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 1].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for py')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.savefig('TrajectoryTracing_py')
plt.show()
plt.close()

# plt.subplot(313)
plt.plot(output_data[1::, 2].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 2].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for pz')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.savefig('TrajectoryTracing_pz')
plt.show()
plt.close()


print(output_data_angle[1::, 3])
plt.plot(output_data_angle[1::, 3].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 3].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for angle_x')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.show()
plt.close()


# plt.subplot(312)
plt.plot(output_data_angle[1::, 4].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 4].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for for angle_y')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.show()
plt.close()


# plt.subplot(313)
plt.plot(output_data_angle[1::, 5].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 5].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for for angle_z')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.show()
plt.close()


#三维画图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(output_data[1::, 0].cpu().numpy(), output_data[1::, 1].cpu().numpy(), output_data[1::, 2].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
ax.plot(predictions[:, 0].cpu().numpy(), predictions[:, 1].cpu().numpy(), predictions[:, 2].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)

# 设置图例和标签
ax.legend()
ax.set_title('Output and Predictions')
ax.set_xlabel('px')
ax.set_ylabel('py')
ax.set_zlabel('pz')

# 显示图形
plt.show()

#三维画图2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(output_data_angle[1::, 3].cpu().numpy(), output_data_angle[1::, 4].cpu().numpy(), output_data_angle[1::, 5].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
ax.plot(predictions[:, 3].cpu().numpy(), predictions[:, 4].cpu().numpy(), predictions[:, 5].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)

# 设置图例和标签
ax.legend()
ax.set_title('Output and Predictions')
ax.set_xlabel('phix')
ax.set_ylabel('phiy')
ax.set_zlabel('phiz')

# 显示图形
plt.show()



# 计算绝对误差
print("轨迹位移预测RMSE:")
print((loss_func(predictions[:, 0:3], output_data[1::, 0:3])).data.cpu())

print("轨迹角度预测RMSE:")
print((loss_func(predictions[:, 3:6], output_data_angle[1::, 3:6])).data.cpu())

print("px")
print((loss_func(predictions[:, 0], output_data[1::, 0])).data.cpu())

print("py")
print((loss_func(predictions[:, 1], output_data[1::, 1])).data.cpu())

print("pz")
print((loss_func(predictions[:, 2], output_data[1::, 2])).data.cpu())

# print("sx")
# print((loss_func(predictions[:, 3], output_data[1::, 3])).data.cpu())

# print("sy")
# print((loss_func(predictions[:, 4], output_data[1::, 4])).data.cpu())

# print("sz")
# print((loss_func(predictions[:, 5], output_data[1::, 5])).data.cpu())

# print("thd")
# print((loss_func(predictions[:, 6], output_data[1::, 6])).data.cpu())

print("angle_x")
print((loss_func(predictions[:, 3], output_data_angle[1::, 3])).data.cpu())

print("angle_y")
print((loss_func(predictions[:, 4], output_data_angle[1::, 4])).data.cpu())

print("angle_z")
print((loss_func(predictions[:, 5], output_data_angle[1::, 5])).data.cpu())


# 计算相对误差并输出
def relative_error(pred, true):
    return torch.abs(pred - true) / torch.abs(true)

# px
px_rel_error = relative_error(predictions[:, 0], output_data[1::, 0])
print("px相对误差平均值:")
print(px_rel_error.mean().data.cpu())

# py
py_rel_error = relative_error(predictions[:, 1], output_data[1::, 1])
print("py相对误差平均值:")
print(py_rel_error.mean().data.cpu())

# pz
pz_rel_error = relative_error(predictions[:, 2], output_data[1::, 2])
print("pz相对误差平均值:")
print(pz_rel_error.mean().data.cpu())

# angle_x
angle_x_rel_error = relative_error(predictions[:, 3], output_data_angle[1::, 3])
print("angle_x相对误差平均值:")
print(angle_x_rel_error.mean().data.cpu())

# angle_y
angle_y_rel_error = relative_error(predictions[:, 4], output_data_angle[1::, 4])
print("angle_y相对误差平均值:")
print(angle_y_rel_error.mean().data.cpu())

# angle_z
angle_z_rel_error = relative_error(predictions[:, 5], output_data_angle[1::, 5])
print("angle_z相对误差平均值:")
print(angle_z_rel_error.mean().data.cpu())

print("轨迹位移预测相对误差:")
p_rel_error = relative_error(predictions[:, 0:3], output_data[1::, 0:3])
print("p相对误差平均值:")
print(p_rel_error.mean().data.cpu())

print("轨迹角度预测相对误差:")
angle_rel_error = relative_error(predictions[:, 3:6], output_data_angle[1::, 3:6])
print("angle相对误差平均值:")
print(angle_rel_error.mean().data.cpu())


def calculate_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

rmse_value = calculate_rmse(predictions[:, 0:3].cpu().numpy(), output_data[1:, 0:3].cpu().numpy())
rmse_value_angle = calculate_rmse(predictions[:, 3:6].cpu().numpy(), output_data_angle[1:, 3:6].cpu().numpy())

print("RMSE for angles (phix, phiy, phiz):", rmse_value_angle)
print("RMSE for p (px, py, pz):", rmse_value)