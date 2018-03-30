from numpy import *    

def autoNorm(dataSet):  
    minVals = dataSet.min(0)  
    maxVals = dataSet.max(0)  
    range = maxVals - minVals  
    normdataSet = zeros(shape(dataSet))  
    n = dataSet.shape[0]  
    normdataSet = (dataSet - tile(minVals,(n,1)))/tile(range,(n,1))  
    return  normdataSet
    
def sample_autoNorm(sample,train_data):  
    minVals = dataSet.min(0)  
    maxVals = dataSet.max(0)  
    range = maxVals - minVals  
    normdataSet = zeros(shape(dataSet))  
    n = dataSet.shape[0]  
    normdataSet = (dataSet - tile(minVals,(n,1)))/tile(range,(n,1))  
    return  normdataSet
