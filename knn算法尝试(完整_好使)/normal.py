from numpy import *    

def autoNorm(dataSet):  
    minVals = dataSet.min(0)  
    maxVals = dataSet.max(0)  
    allrange = maxVals - minVals  
    #print("跨度大小："+repr(allrange))  
    normdataSet = zeros(shape(dataSet))  
    n = dataSet.shape[0]
    #print("正则化的数据大小："+repr(n))  
    normdataSet = (dataSet - tile(minVals,(n,1)))/tile(allrange,(n,1))  
    return  normdataSet
    
