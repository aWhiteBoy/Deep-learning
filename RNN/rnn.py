#coding=gbk

import numpy as np
from activators import ReluActivator, IdentityActivator
import pdb

#对numpy数组进行按元素操作，并将返回值写回到数组中
def element_wise_op(array, op):
    for i in np.nditer(array,op_flags=['readwrite']):
        i[...] = op(i)

#构建一个循环层的RecurrentLayer类
class RecurrentLayer(object):
	#__init__(self,属性1，属性2....)：类里面的_init_方法，self代表实例，通过self访问实例对象的变量和函数
    #state_width表示隐藏层的宽度
    #初始化_init_方法
    def __init__(self, input_width, state_width,
                 activator, learning_rate):
		#定义self实例的各个属性
		#输入向量x的个数，该个数代表输入特征的维数
        self.input_width = input_width
        #输给隐藏层的s向量的个数，该个数代表隐藏层的宽度
        self.state_width = state_width
        self.activator = activator #？？？？？
        #学习速率
        self.learning_rate = learning_rate
        self.times = 0       # 当前时刻初始化为t0
        self.state_list = [] # 保存各个时刻的state
        #把s向量保存在一起，个数代表隐藏层宽度
        self.state_list.append(np.zeros(
            (state_width, 1)))           # 初始化s0，因为s是个一列的列向量
        self.U = np.random.uniform(-1e-4, 1e-4,
            (state_width, input_width))  # 初始化U  U的列要等于x的行，U的行是输出s的列
        self.W = np.random.uniform(-1e-4, 1e-4,
            (state_width, state_width))  # 初始化W  W的列等于s的行，W的行是输出s的列
            
    #前向传播中实现循环层前向计算   
    def forward(self, input_array):
        '''
        根据s计算公式进行前向计算
        '''
        self.times += 1#前向传播一次，时间步+1
        #计算当前时间步的s
        state = (np.dot(self.U, input_array) +
                 np.dot(self.W, self.state_list[-1]))#前向计算公式实现
        element_wise_op(state, self.activator.forward)#？？？？？？？？
        self.state_list.append(state) #存储新的时间步状态

	#反相传播传递误差项
	#sensitivity_array, activator？？？？？？？？？？？？？？？？？
    def backward(self, sensitivity_array, activator):
        '''
        实现BPTT算法
        '''
        #计算权重的误差项δ
        self.calc_delta(sensitivity_array, activator)
        #权重梯度的计算
        self.calc_gradient()    

    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        self.W -= self.learning_rate * self.gradient#新权重=原权重-学习率*权重梯度
        
    #计算权重的误差项δ
    def calc_delta(self, sensitivity_array, activator):
        self.delta_list = []  # 用来保存各个时刻的误差项
        for i in range(self.times):
            self.delta_list.append(np.zeros(
                (self.state_width, 1)))
        self.delta_list.append(sensitivity_array)
        # 迭代计算每个时刻的误差项
        for k in range(self.times - 1, 0, -1):
            self.calc_delta_k(k, activator)
            
    def calc_delta_k(self, k, activator):
        '''
        根据k+1时刻的delta计算k时刻的delta
        '''
        state = self.state_list[k+1].copy()
        element_wise_op(self.state_list[k+1],
                    activator.backward)
        self.delta_list[k] = np.dot(
            np.dot(self.delta_list[k+1].T, self.W),
            np.diag(state[:,0])).T
            
    def calc_gradient(self):
        self.gradient_list = [] # 保存各个时刻的权重梯度
        for t in range(self.times + 1):
            self.gradient_list.append(np.zeros((self.state_width, self.state_width)))
        for t in range(self.times, 0, -1):
            self.calc_gradient_t(t)
        # 实际的梯度是各个时刻梯度之和
        self.gradient = reduce(
            lambda a, b: a + b, self.gradient_list,
            self.gradient_list[0]) # [0]被初始化为0且没有被修改过
            
    def calc_gradient_t(self, t):
        '''
        计算每个时刻t权重的梯度
        '''
        gradient = np.dot(self.delta_list[t],
            self.state_list[t-1].T)
        self.gradient_list[t] = gradient
        
    def reset_state(self):
        self.times = 0       # 当前时刻初始化为t0
        self.state_list = [] # 保存各个时刻的state
        self.state_list.append(np.zeros(
            (self.state_width, 1)))      # 初始化s0
            
def data_set():
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d
    
def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    
    rl = RecurrentLayer(3, 2, IdentityActivator(), 1e-3)

    # 计算forward值
    x, d = data_set()
    rl.forward(x[0])
    rl.forward(x[1])
    
    # 求取sensitivity map
    sensitivity_array = np.ones(rl.state_list[-1].shape,
                                dtype=np.float64)
    # 计算梯度
    rl.backward(sensitivity_array, IdentityActivator())
    
    # 检查梯度
    epsilon = 10e-4
    for i in range(rl.W.shape[0]):
        for j in range(rl.W.shape[1]):
            rl.W[i,j] += epsilon
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            err1 = error_function(rl.state_list[-1])
            rl.W[i,j] -= 2*epsilon
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            err2 = error_function(rl.state_list[-1])
            expect_grad = (err1 - err2) / (2 * epsilon)
            rl.W[i,j] += epsilon
            print ('weights(%d,%d): expected - actural %f - %f' % (
                i, j, expect_grad, rl.gradient[i,j]))

def test():
    #l = RecurrentLayer(3, 2, ReluActivator(), 1e-3)
    x, d = data_set()
    #pdb.set_trace()
    #l.forward(x[0])
    #l.forward(x[1])
    #l.backward(d, ReluActivator())
    #return l

#test()
