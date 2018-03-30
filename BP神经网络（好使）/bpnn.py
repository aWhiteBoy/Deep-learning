import math
import random
import pdb
from numpy import *
import read_csv
import matplotlib.pyplot as plt
import pdb  

#撒下随机数种子
random.seed(0)

def rand(a, b):
    return (b - a) * random.random() + a

#创造一个指定大小的矩阵
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
        
    #print(type(mat))
    #pdb.set_trace()
    return mat

#定义sigmoid函数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

#sigmod函数的导数
def sigmoid_derivative(x):
    return x * (1 - x)

#标签一维转二维
def change_labels_dimen(new_label,labels):
    for x in range(len(labels)):
        #print(len(labels))
        if labels[x]==1:
           #print(len(labels))
           new_label=row_stack((new_label,array([1,0,0])))
        elif labels[x]==2:
           new_label=row_stack((new_label,[0,1,0]))
        elif labels[x]==3:
           new_label=row_stack((new_label,[0,0,1]))
    new_label=new_label[1:len(labels)+1,:]
    #pdb.set_trace()
    	    
    return  new_label

#定义BPNeuralNetwork类
#使用三个列表维护输入层，隐含层和输出层神经元
#列表中的元素代表对应神经元当前的输出值
#使用两个二维列表以邻接矩阵的形式维护输入层与隐含层
#隐含层与输出层之间的连接权值， 通过同样的形式保存矫正矩阵
class BPNeuralNetwork:
    def __init__(self):#初始化变量  
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []
        
	#定义setup方法初始化神经网络
    def setup(self, ni, nh, no):
        self.input_n = ni + 1#输入层+偏置项(此代码是放在输入层) 
        self.hidden_n = nh#隐含层（此处给了5层） 
        self.output_n = no#输出层（一个）
        # init cells 初始化神经元  
        self.input_cells = [1.0] * self.input_n #[1 1 1 1]
        #print(self.input)
        #print(type(self.input))
        #print(self.input_cells)
        self.hidden_cells = [1.0] * self.hidden_n #[1 1 1 1 1]
        self.output_cells = [1.0] * self.output_n #[1]
        # init weights 初始化连接边的边权
        self.input_weights = make_matrix(self.input_n, self.hidden_n)#邻接矩阵存边权：输入层->隐藏层  行取前面层，列取后面层
        self.output_weights = make_matrix(self.hidden_n, self.output_n)#邻接矩阵存边权：隐藏层->输出层
        # random activate
        #随机初始化边权：为了反向传导做准备--->随机初始化的目的是使对称失效 
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)#由输入层第i个元素到隐藏层第j个元素的边权为随机值 
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)#由隐藏层第i个元素到输出层第j个元素的边权为随机值
        # init correction matrix#保存校正矩阵，为了以后误差做调整 
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        #权重矩阵存历史值
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

	#定义predict方法进行一次前馈,输出预测值
    def predict(self, inputs):
        #把输入层转到self变量中
        #pdb.set_trace()	
        for i in range(self.input_n - 1):#除去自加的偏置层
            self.input_cells[i] = inputs[i]
        #计算隐藏层的输出，每个节点最终的输出值就是权值*节点值的加权和 
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
				#输入层的行*隐藏层的列
                #pdb.set_trace()	
                total += self.input_cells[i] * self.input_weights[i][j]
                #pdb.set_trace()	
            self.hidden_cells[j] = sigmoid(total)
        # 计算输出层的输出
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
                #此处为何是先i再j，以隐含层节点做大循环，输入样本为小循环，是为了每一个隐藏节点计算一个输出值，传输到下一层 
            self.output_cells[k] = sigmoid(total)#获取输出层每个元素的值  
            #此节点的输出是前一层所有输入点和到该点之间的权值加权和
        return self.output_cells[:]#最后输出层的结果返回 

	#定义back_propagate方法定义一次反向传播和更新权值的过程， 并返回最终预测误差:
	#反向传播算法：调用预测函数，根据反向传播获取权重后前向预测，将结果与实际结果返回比较误差  
    def back_propagate(self, case, label, learn, correct):
        # feed forward
        #对输入样本做预测 
        self.predict(case)#对实例进行预测 
        # get output layer error
        output_deltas = [0.0] * self.output_n#初始化矩阵  
        for o in range(self.output_n):
            #pdb.set_trace()
            error = label[o] - self.output_cells[o]#正确结果和预测结果的误差：0,1，-1 
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
            #误差稳定在0~1内  
        # get hidden layer error
        #隐含层误差  
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] *  self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights
        #反向传播算法求W  
        #更新隐藏层->输出权重 
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                #调整权重：上一层每个节点的权重学习*变化+矫正率
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        #更新输入->隐藏层的权重 
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        #获取全局误差 
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2#平方误差函数  
        return error

	#定义train方法控制迭代， 该方法可以修改最大迭代次数， 学习率λ， 矫正率μ三个参数.
    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        #pdb.set_trace()
        for j in range(limit):#设置迭代次数  
            error = 0.0
            for i in range(len(cases)):#对输入层进行访问  
                label = labels[i]
                #pdb.set_trace()
                case = cases[i]
                #pdb.set_trace()
                error += self.back_propagate(case, label, learn, correct)  

	#编写test方法，演示如何使用神经网络学习异或逻辑:
    def test(self):
        i=0
        num=0
        for case in testset:
            if self.predict(case)[0]>=0.8 :
               case=[1,0,0]
            elif self.predict(case)[1]>=0.8 :
               case=[0,1,0]
            elif self.predict(case)[2]>=0.8 :
               case=[0,0,1]
            if case==new_test_label[i]:
               num=num+1
            i=i+1
        print(num)
        print(len(new_test_label))
        rate=float("%.2f" %((float(num)/len(new_test_label))*100))
        print("\n正确率是："+repr(rate)+"%")	
       
	#编写train方法
    def train_data(self):
        cases = trainset
        labels =new_train_label
        #pdb.set_trace()
        self.setup(4, 5, 3)#初始化神经网络：输入层，隐藏层，输出层元素个数  
        #pdb.set_trace()
        self.train(cases, labels, 1000, 0.05, 0.1)

filename = 'C:/Users\LY_BOY/Desktop/BP神经网络（好使）/iris.csv'
trainset = [[0 for col in range(4)] for row in range(1)]
trainnum=0
testnum=0
testset = [[0 for col in range(4)] for row in range(1)]
train_label=[]
test_label=[]
new_train_label = [[0 for col in range(3)] for row in range(1)]
new_test_label = [[0 for col in range(3)] for row in range(1)]

nn = BPNeuralNetwork()
trainset,testset,train_label,test_label=read_csv.displaydata(filename,0.67,trainset,testset,train_label,test_label,testnum,trainnum)
new_train_label=change_labels_dimen(new_train_label,train_label)
#pdb.set_trace()
new_test_label=change_labels_dimen(new_test_label,test_label)
trainset=trainset.tolist()
print(trainset)
print(type(trainset))
new_train_label=new_train_label.tolist()

testset=testset.tolist()
new_test_label=new_test_label.tolist()

print(new_train_label)
print(type(new_train_label))
#pdb.set_trace()
nn.train_data()
nn.test()

#数据可视化
plt.show()	
