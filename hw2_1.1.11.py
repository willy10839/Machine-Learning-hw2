import numpy as np
from numpy.linalg import pinv,inv,matrix_power
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math
import csv
import random
input_data=[]
target_data=[]
data=[]
s=0.1
with open('1_data.csv', newline='') as csvfile:
    train_data = csv.reader(csvfile)
    next(train_data, None)  # skip the headers
    for i in train_data:
        i = [float(j) for j in i]
        data.append(i)
        input_data.append(i[0])
        target_data.append(i[1])
#print(data)
'''result_dict=loadmat("1_data")
x = loadmat("1_data")['x']
t = loadmat("1_data")['t']
s=0.1
x_t=np.concatenate((x,t),axis=1)
data_sort=sorted(x_t, key=lambda k: k[0], reverse=True)


print(x_t)
target_data=np.array(result_dict['t'])
input_data=np.array(result_dict['x'])'''

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def cal(data,N):
    x_100=[]
    for i in range(len(data)):
        x_100.append(data[i][0])
    x_100=sorted(x_100)
    tmp_data=[]
    for i in range(N):
        tmp_data.append(data[i])
    #tmp_data=sorted(tmp_data, key=lambda k: k[0], reverse=True)
    xx=[]
    tt=[]
    for i in range(N):
        xx.append(tmp_data[i][0])
        tt.append(tmp_data[i][1])
    target_tmp=[]
    for i in range(N):
        target_tmp.append(tmp_data[i][1])
    input_seven_dim=[]
    for i in range(N):
        tmp=[]
        for j in range(7):
            u=4*j/7
            x=(tmp_data[i][0]-u)/s
            #out=1/(1+np.exp(-x))
            tmp.append(sigmoid(x))
        input_seven_dim.append(tmp)
    mat=np.array(input_seven_dim)#10*7
    matrix=mat.T.dot(mat)# 7*10*10*7
    covariance_inverse=0.000001*np.identity(7)+matrix
    covariance=np.linalg.inv(covariance_inverse) #7*7
    mean=covariance.dot(mat.T.dot(target_tmp))
    weight_array = np.random.multivariate_normal(mean, covariance, 5)
    curve=[]
    for i in range(len(data)):
        curve_tmp=[]
        for j in range(7):
            u=4*j/7
            x=(x_100[i]-u)/s
            curve_tmp.append(sigmoid(x))
        curve.append(curve_tmp)
    curve=np.array(curve)
    for i in range(5):
         y_func = np.dot(curve,weight_array[i])
         #print(y_func)
         #print(weight_array[i])
         #print(curve.shape)
         plt.plot(x_100,y_func, 'r--')
    plt.plot(xx,target_tmp,'mo')
    plt.show()
    mean_array=[]
    for i in range(100):
        mean_array.append(mean.dot(curve[i].T))
    variance=[]
    for i in range(100):
        variance.append( curve[i].dot(covariance).dot(curve[i].T))
    sq=np.sqrt(variance)
    plt.plot(x_100, mean_array, 'r--')
    plt.plot(xx, tt , 'bo')
    plt.fill_between(x_100,  mean_array-sq,  mean_array+sq, facecolor='#ffc67c')
    plt.show()
    print("The mean of N= ",N," is :",mean)
    print("The covariance of N= ",N," is :",covariance)

cal(data,10)
cal(data,15)
cal(data,30)
cal(data,80)

