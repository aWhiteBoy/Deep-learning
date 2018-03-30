# coding=gbk
import matplotlib.pyplot as plt
import numpy as np

def draw_data(data,labels):
	plt.figure()
	plt.subplot(2,2,1)#����һ���������еĻ�������һ��
	x = list(labels)
	print("\n����x"+repr(x)+"\n����:"+repr(len(x)))
	y = list(data[:,0])
	print("\n����y"+repr(y)+"\n����:"+repr(len(y)))
	plt.xlabel("labels")
	plt.ylabel("Sepal.Length")
	#sָ���С��alphaָ͸����
	plt.scatter(x,y,s=8,color='blue',alpha=0.5)
	plt.subplot(2,2,2)#�ڶ���
	x = list(labels)
	print("\n����x"+repr(x)+"\n����:"+repr(len(x)))
	y = list(data[:,1])
	print("\n����y"+repr(y)+"\n����:"+repr(len(y)))
	plt.xlabel("labels")
	plt.ylabel("Sepal.Width")
	#sָ���С��alphaָ͸����
	plt.scatter(x,y,s=8,color='red',alpha=0.5)
	plt.subplot(2,2,3)#������
	x = list(labels)
	print("\n����x"+repr(x)+"\n����:"+repr(len(x)))
	y = list(data[:,2])
	print("\n����y"+repr(y)+"\n����:"+repr(len(y)))
	plt.xlabel("labels")
	plt.ylabel("Petal.Length")
	#sָ���С��alphaָ͸����
	plt.scatter(x,y,s=8,color='black',alpha=0.5)
	plt.subplot(2,2,4)#���ĸ�
	x = list(labels)
	print("\n����x"+repr(x)+"\n����:"+repr(len(x)))
	y = list(data[:,3])
	print("\n����y"+repr(y)+"\n����:"+repr(len(y)))
	plt.xlabel("labels")
	plt.ylabel("Petal.Width")
	#sָ���С��alphaָ͸����
	plt.scatter(x,y,s=8,color='green',alpha=0.5)
	
