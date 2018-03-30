# coding=gbk
from numpy import *  
import pdb
import normal

#求距离
#输入样本特征矩阵（某一个的），测试用例特征
def getDist(dataSet,Sample): 
    #print("特征"+repr(dataSet)) 
    #print("样本"+repr(Sample))
    #print("作差"+repr(dataSet - Sample))
    #print("误差和"+repr(sum(power(dataSet - Sample,2)) ))
    return sum(power(dataSet - Sample,2)) 
   
#分析距离，距离和标签顺序排序
def sortDist(dist,labels,train_data):
	#求距离数组的大小  
    n = shape(dist)[0]  
    for i in range(1,n):  
        for j in range(n-i):  
            if (dist[j] > dist[j+1]): 
                #pdb.set_trace() 
                train_data[[j,j+1],:] = train_data[[j+1,j],:]     
                #pdb.set_trace()
                temp1 = dist[j]  
                #pdb.set_trace() 
                dist[j] = dist[j+1]  
                dist[j+1] = temp1  
                #pdb.set_trace() 
                temp2 = labels[j]  
                labels[j] = labels[j+1]  
                labels[j+1] = temp2                    
    return dist,labels,train_data  

#计算最佳分类结果
#n为标签总数
def countLabels(labels,k,n):  
    labelCount = zeros(n) 
    for i in range(k):  
        labelCount[labels[i]-1] = labelCount[labels[i]-1]+ 1  
    maxcount = -1  
    for i in range(n): 
		#可能产生误差 
        if(labelCount[i] > maxcount):  
            maxcount = labelCount[i]
            label = i  
    return label+1
    
#knn结果
#sample为测试用例
def doKnn(Sample,dataSet,labels,k,label_num): 
	#n是样本数量，d是特征数量 
    n,d = dataSet.shape
    #pdb.set_trace()
    dist = zeros(n) #建立元素个数为n的一维0矩阵  
    #dist里有测试例子距所有的点的欧式距离
    #Sample= normal.autoNorm(Sample)
    for i in range(n):  
        dist[i] = getDist(dataSet[i],Sample)
        #print("逐个距离:"+repr(dist[i]))  
    #pdb.set_trace()  
    dist,labels = sortDist(dist,labels)
    #print(dist)  
    #pdb.set_trace()  
    label = countLabels(labels,k,label_num)
    
    
    if(label==1):
       label="setosa"
    elif(label==2):
       label="versicolor"
    elif(label==3):
       label="virginica"
    else:
       label="error!!!"		      
		
    print ("\n分类结果是：",end="")
    print (label,"\n") 
    
#knn结果
#sample为测试用例
def doKnn_work(Sample,dataSet,labels,k,label_num): 
	#n是样本数量，d是特征数量 
    n,d = dataSet.shape
    #pdb.set_trace()
    dist = zeros(n) #建立元素个数为n的一维0矩阵  
    #dist里有测试例子距所有的点的欧式距离
    for i in range(n):  
        dist[i] = getDist(dataSet[i],Sample)  
    #pdb.set_trace()  
    #print("排序前data："+repr(dataSet)) 
    #print("排序前距离："+repr(dist))  
    #print("排序前分类："+repr(labels)) 
    #pdb.set_trace()  
    dist,labels,dataSet = sortDist(dist,labels,dataSet)  
    #pdb.set_trace()  
    #print("排序后data："+repr(dataSet)) 
    #pdb.set_trace()  
    #print("排序后距离："+repr(dist))  
    #print("排序后分类："+repr(labels))  
    label = countLabels(labels,k,label_num)
    #print("排序算出结果"+repr(label))  
    
    return label

#测试样本正确率
def Knn_realrate(traindata,trainlabels,testdata,testlabels,k,label_num): 
	num=0;
	for i in range(len(testdata)):
		#print("现在的testdata是:"+repr(traindata))
		if(doKnn_work(testdata[i],traindata,trainlabels,k,label_num)==testlabels[i]):
			num=num+1
		#else:
			#print("错误！")
			#print("第"+repr(i+1)+"个测试集")
			#print("正确答案:"+repr(testlabels[i]))
			#print("计算结果:"+repr(doKnn_work(testdata[i],traindata,trainlabels,k,label_num)))
			
			#print(testlabels[i])
	rate=0.0
	rate=float("%.2f" %((float(num)/len(testdata))*100))
	print("\n正确率是："+repr(rate)+"%")	
	return rate
		




























