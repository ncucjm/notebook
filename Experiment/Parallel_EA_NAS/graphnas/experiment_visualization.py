import os
import json
import matplotlib.pyplot as plt


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

   name = ["", "random_Citeseer_random_none_none_age.txt",
           "random_Citeseer_wheel_1_0.2_noneage.txt",
           "random_Citeseer_wheel_1_0.8_noneage.txt",
           "Citeseer_random_none_none_age.txt",
           "Citeseer_wheel_1_0.2_noneage.txt",
           "Citeseer_wheel_1_0.8_noneage.txt"]


   time_data1, acc_data1 = experiment_data_read(name[1])
   time_data2, acc_data2 = experiment_data_read(name[2])
   time_data3, acc_data3 = experiment_data_read(name[3])

   time_data4, acc_data4 = experiment_data_read(name[4])
   time_data5, acc_data5 = experiment_data_read(name[5])
   time_data6, acc_data6 = experiment_data_read(name[6])



   # 实验1 random+ev2_0.2
   x_rand_1 = [0]
   for index in range(len(time_data1) - 1):
       time_point = x_rand_1[index] + time_data1[index + 1] - time_data1[index]
       x_rand_1.append(int(time_point))

   x_e_1 = [x_rand_1[-1]]
   for index in range(len(time_data4) - 1):
       time_point = x_e_1[-1] + time_data4[index + 1] - time_data4[index]
       x_e_1.append(int(time_point))

   y_rand_1 = [0] + acc_data1
   y_e_1 = [y_rand_1[-1]] + acc_data4

   # 实验2 random+ev2_0.8
   x_rand_2 = [0]
   for index in range(len(time_data2) - 1):
       time_point = x_rand_2[index] + time_data2[index + 1] - time_data2[index]
       x_rand_2.append(int(time_point))

   x_e_2 = [x_rand_2[-1]]
   for index in range(len(time_data5) - 1):
       time_point = x_e_2[-1] + time_data5[index + 1] - time_data5[index]
       x_e_2.append(int(time_point))

   y_rand_2 = [0] + acc_data2
   y_e_2 = [y_rand_2[-1]] + acc_data5

   # 实验3 random+ev1
   x_rand_3 = [0]
   for index in range(len(time_data3) - 1):
       time_point = x_rand_3[index] + time_data3[index + 1] - time_data3[index]
       x_rand_3.append(int(time_point))

   x_e_3 = [x_rand_3[-1]]
   for index in range(len(time_data6) - 1):
       time_point = x_e_3[-1] + time_data6[index + 1] - time_data6[index]
       x_e_3.append(int(time_point))

   y_rand_3 = [0] + acc_data3
   y_e_3 = [y_rand_2[-1]] + acc_data6

   plt.figure(figsize=(6, 4))  # 新建一个图像，设置图像大小

   # # random + ev1
  # x_e_1 = [i for i in range(len(x_e_1))]
   x_rand_1 = [i for i in range(len(x_rand_1))]
   x_e_1 = [i + x_rand_1[-1] for i in range(len(x_e_1))]

   plt.plot(x_e_1, y_e_1, 'go-', label='evolution_ev1_none')
   plt.plot(x_rand_1, y_rand_1, 'go:', label='random_initialize')
   y_e_1_index = y_e_1.index(max(y_e_1))
   show_max_1 = "random+ev1: " + "[" + str(x_e_1[y_e_1_index]) + " " + str(y_e_1[y_e_1_index]) + "]"

   # random + ev2_0.2
   x_e_2 = x_e_1
   x_rand_2 = x_rand_1

   plt.plot(x_e_2, y_e_2, 'bo-', label='evolution_ev2_0.2')
   plt.plot(x_rand_2, y_rand_2, 'bo:', label='random_initialize')
   y_e_2_index = y_e_2.index(max(y_e_2))
   show_max_2 = "random+ev2_0.2: " + "[" + str(x_e_2[y_e_2_index]) + " " + str(y_e_2[y_e_2_index]) + "]"

   # random + ev2_0.8
   x_e_3 = x_e_1
   x_rand_3 = x_rand_1

   plt.plot(x_e_3, y_e_3, 'ro-', label='evolution_ev2_0.2')
   plt.plot(x_rand_3, y_rand_3, 'ro:', label='random_initialize')
   y_e_3_index = y_e_3.index(max(y_e_3))
   show_max_3 = "random+ev2_0.8: " + "[" + str(x_e_3[y_e_3_index]) + " " + str(y_e_3[y_e_3_index]) + "]"

   plt.annotate(show_max_1, xytext=(x_e_1[y_e_1_index], y_e_1[y_e_1_index] - 0.05),
                xy=(x_e_1[y_e_1_index], y_e_1[y_e_1_index]))

   plt.annotate(show_max_2, xytext=(x_e_2[y_e_2_index], y_e_2[y_e_2_index]-0.02),
                xy=(x_e_2[y_e_2_index], y_e_2[y_e_2_index]))

   plt.annotate(show_max_3, xytext=(x_e_3[y_e_3_index], y_e_3[y_e_3_index] + 0.03),
                xy=(x_e_3[y_e_3_index], y_e_3[y_e_3_index]))




   plt.title('time-validate_acc', fontsize=20)  # 标题
   plt.xlabel('epoch', fontsize=10)  # x轴标签
   plt.ylabel('validation accurcy', fontsize=10)  # y轴标签
   plt.legend(loc='best')  # 图例
   plt.show()



