import os
import json
import matplotlib.pyplot as plt
import numpy as np

def experiment_data_read(name):
    path = path_get()[1]
    with open(path + "/" + name, "r") as f:
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
   name = ["controler_train.txt", "controler_search.txt",
           "evolution_train_RL_none_age.txt","evolution_train_random_none_age.txt",
           "evolution_train_RL_age.txt","evolution_train_random_age.txt",
           "random_initialize.txt"]
   time_data1, acc_data1 = experiment_data_read(name[0])
   time_data2, acc_data2 = experiment_data_read(name[1])
   time_data3, acc_data3 = experiment_data_read(name[2])
   time_data4, acc_data4 = experiment_data_read(name[3])
   time_data5, acc_data5 = experiment_data_read(name[4])
   time_data6, acc_data6 = experiment_data_read(name[5])
   time_data7, acc_data7 = experiment_data_read(name[6])

    # 实验1 rl+ev2
   x_r_t_1 = [0]
   for index in range(len(time_data1) - 1):
       time_point = x_r_t_1[index] + time_data1[index+1] - time_data1[index]
       x_r_t_1.append(int(time_point))

   x_r_s_1 = [x_r_t_1[-1]]
   for index in range(len(time_data2) - 1):
       time_point = x_r_s_1[-1] + time_data2[index + 1] - time_data2[index]
       x_r_s_1.append(int(time_point))

   x_e_1 = [x_r_s_1[-1]]
   for index in range(len(time_data3) - 1):
       time_point = x_e_1[-1] + time_data3[index + 1] - time_data3[index]
       x_e_1.append(int(time_point))

   y_r_t_1 = [0] + acc_data1
   y_r_s_1 = [y_r_t_1[-1]] + acc_data2
   y_e_1 = [y_r_s_1[-1]] + acc_data3

   # 实验2 random+ev2
   x_rand_2 = [0]
   for index in range(len(time_data7) - 1):
       time_point = x_rand_2[index] + time_data7[index + 1] - time_data7[index]
       x_rand_2.append(int(time_point))

   x_e_2 = [x_rand_2[-1]]
   for index in range(len(time_data4) - 1):
       time_point = x_e_2[-1] + time_data4[index + 1] - time_data4[index]
       x_e_2.append(int(time_point))

   y_rand_2 = [0] + acc_data7
   y_e_2 = [y_rand_2[-1]] + acc_data4

   # 实验3 rl+ev1
   x_r_t_3 = [0]
   for index in range(len(time_data1) - 1):
       time_point = x_r_t_3[index] + time_data1[index + 1] - time_data1[index]
       x_r_t_3.append(int(time_point))

   x_r_s_3 = [x_r_t_3[-1]]
   for index in range(len(time_data2) - 1):
       time_point = x_r_s_3[-1] + time_data2[index + 1] - time_data2[index]
       x_r_s_3.append(int(time_point))

   x_e_3 = [x_r_s_3[-1]]
   for index in range(len(time_data5) - 1):
       time_point = x_e_3[-1] + time_data5[index + 1] - time_data5[index]
       x_e_3.append(int(time_point))

   y_r_t_3 = [0] + acc_data1
   y_r_s_3 = [y_r_t_1[-1]] + acc_data2
   y_e_3 = [y_r_s_1[-1]] + acc_data5

   # 实验4 random+ev1
   x_rand_4 = [0]
   for index in range(len(time_data7) - 1):
       time_point = x_rand_4[index] + time_data7[index + 1] - time_data7[index]
       x_rand_4.append(int(time_point))

   x_e_4 = [x_rand_4[-1]]
   for index in range(len(time_data6) - 1):
       time_point = x_e_4[-1] + time_data6[index + 1] - time_data6[index]
       x_e_4.append(int(time_point))

   y_rand_4 = [0] + acc_data7
   y_e_4 = [y_rand_2[-1]] + acc_data6

   plt.figure(figsize=(6, 4))  # 新建一个图像，设置图像大小

   # rl + ev2
   plt.plot(x_e_1, y_e_1, 'ro-', label='evolution_ev2')
   plt.plot(x_r_s_1, y_r_s_1, 'ro--', label='RL_search_seeds')
   plt.plot(x_r_t_1, y_r_t_1, 'ro:', label='RL_train')
   y_e_1_index = y_e_1.index(max(y_e_1))
   show_max_1 = "RL+ev2: " + "[" + str(x_e_1[y_e_1_index]) + " " + str(y_e_1[y_e_1_index]) + "]"

   # random + ev2
   plt.plot(x_e_2, y_e_2, 'bo-', label='evolution_ev2')
   plt.plot(x_rand_2, y_rand_2, 'bo:', label='random_initialize')
   y_e_2_index = y_e_2.index(max(y_e_2))
   show_max_2 = "random+ev2: " + "[" + str(x_e_2[y_e_2_index]) + " " + str(y_e_2[y_e_2_index]) + "]"

   # # rl + ev1
   plt.plot(x_e_3, y_e_3, 'go-', label='evolution_ev1')
   plt.plot(x_r_s_3, y_r_s_3, 'ro--', label='RL_search_seeds')
   plt.plot(x_r_t_3, y_r_t_3, 'ro:', label='RL_train')
   y_e_3_index = y_e_3.index(max(y_e_3))
   show_max_3 = "RL+ev1: " + "[" + str(x_e_3[y_e_3_index]) + " " + str(y_e_3[y_e_3_index]) + "]"

   # # random + ev1
   plt.plot(x_e_4, y_e_4, 'yo-', label='evolution_ev1')
   plt.plot(x_rand_4, y_rand_4, 'bo:', label='random_initialize')
   y_e_4_index = y_e_4.index(max(y_e_4))
   show_max_4 = "random+ev1: " + "[" + str(x_e_4[y_e_4_index]) + " " + str(y_e_4[y_e_4_index]) + "]"

   plt.annotate(show_max_1, xytext=(x_e_1[y_e_1_index], y_e_1[y_e_1_index]+0.05),
                xy=(x_e_1[y_e_1_index], y_e_1[y_e_1_index]))
   plt.annotate(show_max_2, xytext=(x_e_2[y_e_2_index], y_e_2[y_e_2_index]+0.03),
                xy=(x_e_2[y_e_2_index], y_e_2[y_e_2_index]))
   plt.annotate(show_max_3, xytext=(x_e_3[y_e_3_index], y_e_3[y_e_3_index]-0.05),
                xy=(x_e_3[y_e_3_index], y_e_3[y_e_3_index]))
   plt.annotate(show_max_4, xytext=(x_e_4[y_e_4_index], y_e_4[y_e_4_index]-0.05),
                xy=(x_e_4[y_e_4_index], y_e_4[y_e_4_index]))


   plt.title('time-acc', fontsize=20)  # 标题
   plt.xlabel('second', fontsize=10)  # x轴标签
   plt.ylabel('validation accurcy', fontsize=10)  # y轴标签
   plt.legend(loc='best')  # 图例
   plt.show()



