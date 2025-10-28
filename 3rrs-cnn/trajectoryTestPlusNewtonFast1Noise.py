from mlp2 import MLPNN
import torch
from scipy.io import loadmat
import torch.nn as nn 
import matplotlib.pyplot as plt
import autograd.numpy as np
import math
import time
# from jax import grad
from autograd import jacobian

def TargetFunction(x, theta1_value, theta2_value, theta3_value):
        
    
        r_up = 275
        r_down = 550


        phiz = np.arctan2((-np.sin(x[0]) * np.sin(x[1])), (np.cos(x[0]) + np.cos(x[1])))

        ux = np.cos(x[1]) * np.cos(phiz)
        uy = np.cos(phiz) * np.sin(x[0]) * np.sin(x[1]) + np.cos(x[0]) * np.sin(phiz)
        uz = np.sin(x[0]) * np.sin(phiz) - np.cos(x[0]) * np.cos(phiz) * np.sin(x[1])
        vx = - np.cos(x[1]) * np.sin(phiz)
        vy = np.cos(x[0]) * np.cos(phiz) - np.sin(x[0]) * np.sin(x[1]) * np.sin(phiz)
        vz = np.sin(x[0]) * np.cos(phiz) + np.cos(x[0]) * np.sin(x[1]) * np.sin(phiz)
        wx = np.sin(x[1])
        wy = -np.sin(x[0]) * np.cos(x[1])
        wz = np.cos(x[0]) * np.cos(x[1])

        px = 0.5 * r_up*(ux - vy)
        py = - uy * r_up

        O13x = px + r_up * ux
        O13y = py + r_up * uy
        O13z = x[2] + r_up * uz

        O23x = px - (0.5 * r_up * ux) + (0.5 * np.sqrt(3) * r_up * vx)
        O23y = py - (0.5 * r_up * uy) + (0.5 * np.sqrt(3) * r_up * vy)
        O23z = x[2] - (0.5 * r_up * uz) + (0.5 * np.sqrt(3) * r_up * vz)

        O33x = px - (0.5 * r_up * ux) - (0.5 * np.sqrt(3) * r_up * vx)
        O33y = py - (0.5 * r_up * uy) - (0.5 * np.sqrt(3) * r_up * vy)
        O33z = x[2] - (0.5 * r_up * uz) - (0.5 * np.sqrt(3) * r_up * vz)

        alpha1 = 0
        alpha2 = 2 * math.pi / 3
        alpha3 = 4 * math.pi / 3

        l1 = 700
        l2 = 775

        A1 = 2 * l1 * np.cos(alpha1) * (r_down * np.cos(alpha1) - O13x)
        A2 = 2 * l1 * np.cos(alpha2) * (r_down * np.cos(alpha2) - O23x)
        A3 = 2 * l1 * np.cos(alpha3) * (r_down * np.cos(alpha3) - O33x)

        B1 = 2 * l1 * O13z * np.cos(alpha1) * np.cos(alpha1)
        B2 = 2 * l1 * O23z * np.cos(alpha2) * np.cos(alpha2)
        B3 = 2 * l1 * O33z * np.cos(alpha3) * np.cos(alpha3)

        C1 = (O13x * O13x) - (2 * r_down * O13x * np.cos(alpha1)) + np.cos(alpha1) * np.cos(alpha1) * (r_down * r_down + l1 * l1 - l2 * l2 + O13z * O13z)
        C2 = (O23x * O23x) - (2 * r_down * O23x * np.cos(alpha2)) + np.cos(alpha2) * np.cos(alpha2) * (r_down * r_down + l1 * l1 - l2 * l2 + O23z * O23z)
        C3 = (O33x * O33x) - (2 * r_down * O33x * np.cos(alpha3)) + np.cos(alpha3) * np.cos(alpha3) * (r_down * r_down + l1 * l1 - l2 * l2 + O33z * O33z)

        targetFunc1 = A1 * np.cos(theta1_value) + B1 * np.sin(theta1_value) + C1
        targetFunc2 = A2 * np.cos(theta2_value) + B2 * np.sin(theta2_value) + C2
        targetFunc3 = A3 * np.cos(theta3_value) + B3 * np.sin(theta3_value) + C3
    
        return np.array([targetFunc1, targetFunc2, targetFunc3])



def create_wrapped_TargetFunction(theta1_value, theta2_value, theta3_value):
    def wrapped_TargetFunction(x):
        # x = x._value
        return TargetFunction(x, theta1_value, theta2_value, theta3_value)
    return wrapped_TargetFunction

if_newton = 1

# 定义网络
input_num = 3
hidden1_num = 64
hidden2_num = 128
hidden3_num = 256
hidden4_num = 512
hidden5_num = 1024
hidden6_num = 512
hidden7_num = 256
hidden8_num = 128
hidden9_num = 64
output_num = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#读取网络模型
num_nets = 8
models =[]

for i in range(num_nets):
    model_path = f'./net_cluster{i+1}_6.pkl'  
    model = MLPNN(input_num, hidden1_num, hidden2_num, hidden3_num, hidden4_num, hidden5_num, hidden6_num, hidden7_num, hidden8_num, hidden9_num, output_num)
    model.load_state_dict(torch.load(model_path))
    models.append(model.to(device))



# 读取数据
data = loadmat('./data/TrajectorySin_0402.mat')
dataset = data['TrajectorySin_0402']

# 设置高斯白噪声
def wgn(x, snr):
    len_x = len(x)
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return x + noise

# plt.plot(dataset[:, 0])
# plt.show()
# plt.close()

snr = 50
dataset[:, 0] = wgn(dataset[:, 0], snr)
dataset[:, 1] = wgn(dataset[:, 1], snr)
dataset[:, 2] = wgn(dataset[:, 2], snr)

# plt.plot(dataset[:, 0])
# plt.show()
# plt.close()




# dataset = dataset[155:500]

input_data = torch.from_numpy(dataset[:, 0:3]).float()
# input_dataset = []
# for scaler_type in scalers:
#     input_data_temp = torch.from_numpy(scaler_type.transform(input_data)).float()
#     input_dataset.append(input_data_temp.to(device))

output_data = torch.from_numpy(dataset[:, 3:9]).float()

input_data = input_data.to(device)
output_data = output_data.to(device)


loss_func = nn.MSELoss()
loss_func = loss_func.to(device)



# 将三个轨迹的参数同时输入到八个网络中
predictions = []

T_record = []
T_record1 = []

first_iteration = True

for step in range(len(input_data)):
    
    min_distance = float('inf')

    if first_iteration:
        # previous_prediction = output_data[0, :]
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
            input_sample = input_data[step, :].reshape(1, 3)
            prediction = model_type(input_sample).reshape(6)


            # 选择标准
            distance = torch.sqrt(loss_func(prediction[0:6], previous_prediction[0:6]))
            

            # # 用output_data判断distance
            # distance = torch.sqrt(loss_func(prediction[0:6], output_data[step, 0:6]))

            if distance < min_distance:
                model_choose = model_num
                min_distance = distance
                selected_prediction = prediction

    if if_newton:
        print("步数：", step)
        DegreeToRad = np.pi/180
        input_sample = input_sample.cpu().numpy()[0]
        selected_prediction = selected_prediction.cpu().numpy()
        [theta1_value, theta2_value, theta3_value] = [input_sample[0] * DegreeToRad, input_sample[1] * DegreeToRad, input_sample[2] * DegreeToRad]


        epsilon = 1e-2
        max_iter = 500
        iter_count = 0

        # 计时
        T1 = time.time()
        
        x_last = np.array([selected_prediction[0] * DegreeToRad, selected_prediction[1] * DegreeToRad, selected_prediction[5]])
        x_cur = x_last


        while iter_count < max_iter:            
            funcs = TargetFunction(x_cur, theta1_value, theta2_value, theta3_value)
            x_norm = np.linalg.norm(funcs.astype('float'), ord=None, axis=None, keepdims=False)
            if(x_norm <= epsilon):
                break
            else:
                custom_wrapped_func = create_wrapped_TargetFunction(theta1_value, theta2_value, theta3_value)
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
        phiz_subs = np.arctan2((-np.sin(x_cur[0]) * np.sin(x_cur[1])), (np.cos(x_cur[0]) + np.cos(x_cur[1])))

        r_up = 275

        ux = np.cos(x_cur[1]) * np.cos(phiz_subs)
        uy = np.cos(phiz_subs) * np.sin(x_cur[0]) * np.sin(x_cur[1]) + np.cos(x_cur[0]) * np.sin(phiz_subs)
        vy = np.cos(x_cur[0]) * np.cos(phiz_subs) - np.sin(x_cur[0]) * np.sin(x_cur[1]) * np.sin(phiz_subs)

        px_subs = 0.5 * r_up*(ux - vy)
        py_subs = - uy * r_up

        selected_prediction[0] = x_cur[0]/np.pi*180
        selected_prediction[1] = x_cur[1]/np.pi*180
        selected_prediction[2] = phiz_subs/np.pi*180
        selected_prediction[3] = px_subs
        selected_prediction[4] = py_subs
        selected_prediction[5] = x_cur[2]

        selected_prediction = torch.from_numpy(selected_prediction).to(device)


    # if step >= 221 and step <= 251:
    #     print("第", step, "步:", "\n", "选择模型:", model_choose, "\n", "牛顿法后的prediction:", selected_prediction, "\n")
    predictions.append(selected_prediction)

predictions = torch.stack(predictions)

print("牛顿法的平均时间：", sum(T_record)/len(T_record))
print("每一步预测的平均时间：", sum(T_record1)/len(T_record1))

# if if_newton:
#     np.save("trace_prediction.npy", predictions.cpu().numpy())
#     np.save("trace_output_data.npy", output_data[1::, :].cpu().numpy())

# 绘图
# plt.figure(figsize=(10, 15))

# plt.subplot(311)
plt.plot(output_data[1::, 0].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 0].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for phix')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.savefig('TrajectoryTracing_phix1')
plt.show()
plt.close()

# plt.subplot(312)
plt.plot(output_data[1::, 1].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 1].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for phiy')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.savefig('TrajectoryTracing_phiy1')
plt.show()
plt.close()

# plt.subplot(313)
plt.plot(output_data[1::, 2].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 2].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for phiz')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.savefig('TrajectoryTracing_phiz1')
plt.show()
plt.close()

# plt.figure(figsize=(10, 15))

# plt.subplot(311)
plt.plot(output_data[1::, 3].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 3].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for px')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.savefig('TrajectoryTracing_px1')
plt.show()
plt.close()


# plt.subplot(312)
plt.plot(output_data[1::, 4].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 4].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for py')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.savefig('TrajectoryTracing_py1')
plt.show()
plt.close()


# plt.subplot(313)
plt.plot(output_data[1::, 5].cpu().numpy(), label='Ground Truth', linestyle='-', marker=' ', linewidth=2.5)
plt.plot(predictions[:, 5].cpu().numpy(), label='Predictions', linestyle='--', marker=' ', linewidth=2.5)
plt.title('Output and Predictions for pz')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()

plt.savefig('TrajectoryTracing_pz1')
plt.show()
plt.close()




# 计算绝对误差
print("轨迹角度预测RMSE:")
print(torch.sqrt(loss_func(predictions[:, 0:3], output_data[1::, 0:3])).data.cpu())

print("轨迹位移预测RMSE:")
print(torch.sqrt(loss_func(predictions[:, 3:6], output_data[1::, 3:6])).data.cpu())

print("phix")
print(torch.sqrt(loss_func(predictions[:, 0], output_data[1::, 0])).data.cpu())

print("phiy")
print(torch.sqrt(loss_func(predictions[:, 1], output_data[1::, 1])).data.cpu())

print("phiz")
print(torch.sqrt(loss_func(predictions[:, 2], output_data[1::, 2])).data.cpu())

print("px")
print(torch.sqrt(loss_func(predictions[:, 3], output_data[1::, 3])).data.cpu())

print("py")
print(torch.sqrt(loss_func(predictions[:, 4], output_data[1::, 4])).data.cpu())

print("pz")
print(torch.sqrt(loss_func(predictions[:, 5], output_data[1::, 5])).data.cpu())


# 计算相对误差
print("位移的绝对误差：")
print(torch.mean(torch.abs((predictions[:, 3:6] - output_data[1::, 3:6]) / output_data[1::, 3:6])).item() * 100, "%")




