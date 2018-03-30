# coding=gbk
import csv
import random
import numpy
import normal
import draw

#从文件中读取数据
#字符串转为整数
#将数据分为训练集和测试集
def displaydata(filename,split,traindata,testdata,traindata_label=[],testdata_label=[],testnum=0,trainnum=0):
	with open(filename) as csvfile:
		lines=csv.reader(csvfile)
		head_row = next(lines)
		dataset=list(lines)
		dataset=numpy.array(dataset)
		data_all_dis=dataset[:,1:5]
		data_labels=dataset[:,5]
		print("\n提取出的原始特征矩阵：\n",data_all_dis)
		print("\n提取出的原始分类：\n",data_labels)
		
		for i in range(150):  
			if(data_labels[i] == "setosa"):  
				data_labels[i] = int('1')
			if(data_labels[i] == "versicolor"): 
				data_labels[i] = int('2')
			if(data_labels[i] == "virginica"): 
				data_labels[i] = int('3')
				
		labels = data_labels
		labels=[int(x) for x in labels]
		print("\n标签特征矩阵:\n",labels) 
		dataSet = [[0] * 4] *150
		for i in range(150):
			dataSet[i]=[float(x) for x in data_all_dis[i]]
		dataSet = numpy.mat(dataSet)   
		draw.draw_data(dataSet,labels)
		print("\n总样本的特征矩阵:\n",dataSet)
		dataSet= normal.autoNorm(dataSet)
		print("\n正则化后的总样本的特征矩阵:\n",dataSet)
		
		for x in range(len(dataSet)):
			if random.random() <= split:
				traindata=numpy.row_stack((traindata,dataSet[x]))
				traindata_label.append(labels[x])
			else:
				testdata=numpy.row_stack((testdata,dataSet[x]))
				testdata_label.append(labels[x])
		
		traindata=traindata[1:len(traindata),:]	
		#traindata=numpy.mat(traindata) 
		#traindata_label=numpy.mat(traindata_label)
		testdata=testdata[1:len(testdata),:]	
		#testdata=numpy.mat(testdata) 	
		#testdata_label=numpy.mat(testdata_label)    
		
		print("\n训练集特征矩阵:\n",traindata)
		print("\n训练集分类矩阵:\n",traindata_label)
		print("\n测试集特征矩阵:\n",testdata)
		print("\n测试集分类矩阵:\n",testdata_label)
		trainnum=len(traindata)
		print("\n训练集数量:\n"+repr(trainnum))
		testnum=len(testdata)
		print("\n测试集数量:\n"+repr(testnum))
		
		return traindata,testdata,traindata_label,testdata_label
		
		
		
			
