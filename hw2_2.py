import numpy as np
import csv
import matplotlib.pyplot as plt
target=[]
feature=[]
w=[]
weight1=np.random.randint(1, size=7)
weight2=np.random.randint(1, size=7)
weight3=np.random.randint(1, size=7)
w.append(weight1)
w.append(weight2)
w.append(weight3)
with open('train.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for i in  rows:
        i = [float(j) for j in i]
        target.append(i[0:3])
        feature.append(i[3:10])

feature_np=np.array(feature)

def softmaxcal(test,w):
    exp=[]
    for i in w:
        exp.append(np.exp(i.dot(test)))
    sum_of_exps=sum(exp)
    softmax = [j/sum_of_exps for j in exp]
    return softmax

def diag(w,matrix_input,n):
    tmp=[]
    for i in range(len(matrix_input)):
        number=softmaxcal(matrix_input[i],w)[n]
        tmp.append(number*(1-number))
    return np.diag(tmp)

def predict(w,matrix_input,n):
    tmp=[]
    for i in range(len(matrix_input)):
        number=softmaxcal(matrix_input[i],w)[n]
        tmp.append(number)
    return np.array(tmp)

def predict_value(w,test,n):
    tmp=softmaxcal(test,w)[n]
    return tmp

def target_num(tar,n):
    tmp=[]
    for i in tar:
        tmp.append(i[n])
    return np.array(tmp)

def update(w,matrix,n):
    tmp=0
    for i in matrix:
        tmp+=predict_value(w,i,n)*(1-predict_value(w,i,n))
    H=tmp*(matrix.T.dot(matrix)).T
    delta=0
    for i in range(len(matrix)):
        delta+=(predict_value(w,matrix[i],n)-target[i][n])*matrix[i]
    H_inverse=np.linalg.inv(H)
    minus=delta.dot(H_inverse)
    return minus

accu=[]
for i in target:
    for j in i:
        if j==1:
            accu.append(i.index(1))
#print(accu)
error_list=[]
count_list=[]
epochs=list(range(1,501))

for i in range(len(epochs)):
    for j in range(3):
        w[j]=w[j]-update(w,feature_np,j)
    error=0
    for k in range(len(feature)):
        tmp=-np.log(np.array(softmaxcal(feature[k],w)))
        error+=tmp.dot(target[k])
    #print(error)
    error_list.append(error)
    count=0
    for i in range(len(feature_np)):
        tmp=softmaxcal(feature_np[i],w)
        tmp_max=max(tmp)
        if accu[i]==tmp.index(tmp_max):
            count+=1
    count_list.append(count)
    #print(count)
print(error_list[-1])
'''error=0
for i in range(len(feature)):
    tmp=-np.log(np.array(softmaxcal(feature[i],w)))
    error+=tmp.dot(target[i])
print(error)'''

test_feature=[]
with open('test.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for i in  rows:
        i = [float(j) for j in i]
        test_feature.append(i[0:7])
test_feature_np=np.array(test_feature)
test_x=list(range(len(test_feature)))
test_y=[]

for i in test_feature_np:
    tmp=softmaxcal(i,w)
    tmp_max=max(tmp)
    test_y.append(tmp.index(tmp_max))
plt.plot(test_x,test_y)
plt.show()

first1_feature=[]
first2_feature=[]
first3_feature=[]
second1_feature=[]
second2_feature=[]
second3_feature=[]
for i in range(len(feature_np)):
    tmp=softmaxcal(feature_np[i],w)
    tmp_max=max(tmp)
    if tmp.index(tmp_max)==0:
        first1_feature.append(feature_np[i][0])
        second1_feature.append(feature_np[i][1])
    elif tmp.index(tmp_max)==1:
        first2_feature.append(feature_np[i][0])
        second2_feature.append(feature_np[i][1])
    else:
        first3_feature.append(feature_np[i][0])
        second3_feature.append(feature_np[i][1])


plt.scatter(first1_feature,second1_feature, color='r', s=25, marker="o")
plt.scatter(first2_feature,second2_feature, color='g', s=25, marker="o")
plt.scatter(first3_feature,second3_feature, color='b', s=25, marker="o")
plt.show()

plt.plot(epochs,error_list)
plt.plot(epochs,count_list)
plt.show()

feature1=[]
for k in range(7):
    ttt=[]
    for i in range(3):
        tmp=[]
        for j in range(len(feature)):
            if target[j].index(1)==i:
                tmp.append(feature[j][k])
        ttt.append(tmp)
    feature1.append(ttt)
plt.hist(feature1[6][0],bins='auto',color='green')#class0
plt.hist(feature1[6][1],bins='auto',color='blue')#class1
plt.hist(feature1[6][2],bins='auto',color='red')#class2
plt.show()

