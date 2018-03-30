# coding=gbk
import numpy
import knn_ex2
import normal  
import pdb  
import csv
import read_csv
import matplotlib.pyplot as plt 

filename = 'C:/Users/LY_BOY/Desktop/knn算法尝试(完整_好使)/iris.csv'
trainset = [[0 for col in range(4)] for row in range(1)]
trainnum=0
testnum=0
testset = [[0 for col in range(4)] for row in range(1)]
train_label=[]
test_label=[]

trainset,testset,train_label,test_label=read_csv.displaydata(filename,0.67,trainset,testset,train_label,test_label,testnum,trainnum)
#print("原始训练标签排序："+repr(train_label))
#print("原始训练特征排序："+repr(trainset))

#测试某一个数据
#暂时不能用，测试样例没有归一化
#knn_ex2.doKnn([6.3,4.4,1.3],trainset,train_label,12,3) 

#测试正确率
right_rate=0.0;
right_rate+=knn_ex2.Knn_realrate(trainset,train_label,testset,test_label,10,3)
#right_rate=float("%.2f" %right_rate)
	
#print("之后标签排序:"+repr(train_label))
#print("之后训练特征排序："+repr(trainset))

#数据可视化
plt.show()	





        

