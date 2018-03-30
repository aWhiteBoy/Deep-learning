# coding=gbk
import matplotlib.pyplot as plt
import numpy as np

def draw_data(data,labels):
	plt.figure()
	plt.subplot(2,2,1)#建立一个两行两列的画布，第一个
	x = list(labels)
	print("\n画布x"+repr(x)+"\n长度:"+repr(len(x)))
	y = list(data[:,0])
	print("\n画布y"+repr(y)+"\n长度:"+repr(len(y)))
	plt.xlabel("labels")
	plt.ylabel("Sepal.Length")
	#s指点大小，alpha指透明度
	plt.scatter(x,y,s=8,color='blue',alpha=0.5)
	plt.subplot(2,2,2)#第二个
	x = list(labels)
	print("\n画布x"+repr(x)+"\n长度:"+repr(len(x)))
	y = list(data[:,1])
	print("\n画布y"+repr(y)+"\n长度:"+repr(len(y)))
	plt.xlabel("labels")
	plt.ylabel("Sepal.Width")
	#s指点大小，alpha指透明度
	plt.scatter(x,y,s=8,color='red',alpha=0.5)
	plt.subplot(2,2,3)#第三个
	x = list(labels)
	print("\n画布x"+repr(x)+"\n长度:"+repr(len(x)))
	y = list(data[:,2])
	print("\n画布y"+repr(y)+"\n长度:"+repr(len(y)))
	plt.xlabel("labels")
	plt.ylabel("Petal.Length")
	#s指点大小，alpha指透明度
	plt.scatter(x,y,s=8,color='black',alpha=0.5)
	plt.subplot(2,2,4)#第四个
	x = list(labels)
	print("\n画布x"+repr(x)+"\n长度:"+repr(len(x)))
	y = list(data[:,3])
	print("\n画布y"+repr(y)+"\n长度:"+repr(len(y)))
	plt.xlabel("labels")
	plt.ylabel("Petal.Width")
	#s指点大小，alpha指透明度
	plt.scatter(x,y,s=8,color='green',alpha=0.5)
	
