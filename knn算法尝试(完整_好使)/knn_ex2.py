# coding=gbk
from numpy import *  
import pdb
import normal

#�����
#����������������ĳһ���ģ���������������
def getDist(dataSet,Sample): 
    #print("����"+repr(dataSet)) 
    #print("����"+repr(Sample))
    #print("����"+repr(dataSet - Sample))
    #print("����"+repr(sum(power(dataSet - Sample,2)) ))
    return sum(power(dataSet - Sample,2)) 
   
#�������룬����ͱ�ǩ˳������
def sortDist(dist,labels,train_data):
	#���������Ĵ�С  
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

#������ѷ�����
#nΪ��ǩ����
def countLabels(labels,k,n):  
    labelCount = zeros(n) 
    for i in range(k):  
        labelCount[labels[i]-1] = labelCount[labels[i]-1]+ 1  
    maxcount = -1  
    for i in range(n): 
		#���ܲ������ 
        if(labelCount[i] > maxcount):  
            maxcount = labelCount[i]
            label = i  
    return label+1
    
#knn���
#sampleΪ��������
def doKnn(Sample,dataSet,labels,k,label_num): 
	#n������������d���������� 
    n,d = dataSet.shape
    #pdb.set_trace()
    dist = zeros(n) #����Ԫ�ظ���Ϊn��һά0����  
    #dist���в������Ӿ����еĵ��ŷʽ����
    #Sample= normal.autoNorm(Sample)
    for i in range(n):  
        dist[i] = getDist(dataSet[i],Sample)
        #print("�������:"+repr(dist[i]))  
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
		
    print ("\n�������ǣ�",end="")
    print (label,"\n") 
    
#knn���
#sampleΪ��������
def doKnn_work(Sample,dataSet,labels,k,label_num): 
	#n������������d���������� 
    n,d = dataSet.shape
    #pdb.set_trace()
    dist = zeros(n) #����Ԫ�ظ���Ϊn��һά0����  
    #dist���в������Ӿ����еĵ��ŷʽ����
    for i in range(n):  
        dist[i] = getDist(dataSet[i],Sample)  
    #pdb.set_trace()  
    #print("����ǰdata��"+repr(dataSet)) 
    #print("����ǰ���룺"+repr(dist))  
    #print("����ǰ���ࣺ"+repr(labels)) 
    #pdb.set_trace()  
    dist,labels,dataSet = sortDist(dist,labels,dataSet)  
    #pdb.set_trace()  
    #print("�����data��"+repr(dataSet)) 
    #pdb.set_trace()  
    #print("�������룺"+repr(dist))  
    #print("�������ࣺ"+repr(labels))  
    label = countLabels(labels,k,label_num)
    #print("����������"+repr(label))  
    
    return label

#����������ȷ��
def Knn_realrate(traindata,trainlabels,testdata,testlabels,k,label_num): 
	num=0;
	for i in range(len(testdata)):
		#print("���ڵ�testdata��:"+repr(traindata))
		if(doKnn_work(testdata[i],traindata,trainlabels,k,label_num)==testlabels[i]):
			num=num+1
		#else:
			#print("����")
			#print("��"+repr(i+1)+"�����Լ�")
			#print("��ȷ��:"+repr(testlabels[i]))
			#print("������:"+repr(doKnn_work(testdata[i],traindata,trainlabels,k,label_num)))
			
			#print(testlabels[i])
	rate=0.0
	rate=float("%.2f" %((float(num)/len(testdata))*100))
	print("\n��ȷ���ǣ�"+repr(rate)+"%")	
	return rate
		




























