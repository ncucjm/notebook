import os
import json
import matplotlib.pyplot as plt

def experiment_data_read(name):
    path = path_get()[1]
    with open(path + "/" +name, "r") as f:
        all_data = f.readlines()
        time_data = all_data[0][:-1]
        acc_data = all_data[1]
        time_data = json.loads(time_data)
        acc_data = json.loads(acc_data)
        return time_data, acc_data

def path_get():
    # 当前文件目录
    current_path = os.path.abspath('')
    # 当前文件夹父目录
    father_path = os.path.abspath(os.path.dirname(current_path))
    # corpus_path = os.path.join(father_path, corpus)
    return father_path, current_path

if __name__=="__main__":
   name = ["controler_train.txt", "controler_search.txt", "evolution_train.txt"]
   time_data1, acc_data1 = experiment_data_read(name[0])
   time_data2, acc_data2 = experiment_data_read(name[1])
   time_data3, acc_data3 = experiment_data_read(name[2])

   x_r_t = [0]
   for index in range(len(time_data1) - 1):
       time_point = x_r_t[index] + time_data1[index+1] - time_data1[index]
       x_r_t.append(int(time_point))

   x_r_s = [x_r_t[-1]]
   for index in range(len(time_data2) - 1):
       time_point = x_r_s[-1] + time_data2[index + 1] - time_data2[index]
       x_r_s.append(int(time_point))

   x_e = [x_r_s[-1]]
   for index in range(len(time_data3) - 1):
       time_point = x_e[-1] + time_data3[index + 1] - time_data3[index]
       x_e.append(int(time_point))

   y_r_t = [0] + acc_data1
   y_r_s = [y_r_t[-1]] + acc_data2
   y_e = [y_r_s[-1]] + acc_data3

   plt.figure(figsize=(6, 4))  # 新建一个图像，设置图像大小

   plt.plot(x_e, y_e, 'ro-', label='evolution1_wheel_search_best_gnn')
   plt.plot(x_r_s, y_r_s, 'ro--', label='RL_search_seeds')
   plt.plot(x_r_t, y_r_t, 'ro:', label='RL_train')
   plt.title('time-acc', fontsize=20)  # 标题
   plt.xlabel('second', fontsize=10)  # x轴标签
   plt.ylabel('validation accurcy', fontsize=10)  # y轴标签
   plt.legend(loc='best')  # 图例
   plt.show()



