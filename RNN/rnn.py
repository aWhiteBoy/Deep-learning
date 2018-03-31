#coding=gbk

import numpy as np
from activators import ReluActivator, IdentityActivator
import pdb

#��numpy������а�Ԫ�ز�������������ֵд�ص�������
def element_wise_op(array, op):
    for i in np.nditer(array,op_flags=['readwrite']):
        i[...] = op(i)

#����һ��ѭ�����RecurrentLayer��
class RecurrentLayer(object):
	#__init__(self,����1������2....)���������_init_������self����ʵ����ͨ��self����ʵ������ı����ͺ���
    #state_width��ʾ���ز�Ŀ��
    #��ʼ��_init_����
    def __init__(self, input_width, state_width,
                 activator, learning_rate):
		#����selfʵ���ĸ�������
		#��������x�ĸ������ø�����������������ά��
        self.input_width = input_width
        #������ز��s�����ĸ������ø����������ز�Ŀ��
        self.state_width = state_width
        self.activator = activator #����������
        #ѧϰ����
        self.learning_rate = learning_rate
        self.times = 0       # ��ǰʱ�̳�ʼ��Ϊt0
        self.state_list = [] # �������ʱ�̵�state
        #��s����������һ�𣬸����������ز���
        self.state_list.append(np.zeros(
            (state_width, 1)))           # ��ʼ��s0����Ϊs�Ǹ�һ�е�������
        self.U = np.random.uniform(-1e-4, 1e-4,
            (state_width, input_width))  # ��ʼ��U  U����Ҫ����x���У�U���������s����
        self.W = np.random.uniform(-1e-4, 1e-4,
            (state_width, state_width))  # ��ʼ��W  W���е���s���У�W���������s����
            
    #ǰ�򴫲���ʵ��ѭ����ǰ�����   
    def forward(self, input_array):
        '''
        ����s���㹫ʽ����ǰ�����
        '''
        self.times += 1#ǰ�򴫲�һ�Σ�ʱ�䲽+1
        #���㵱ǰʱ�䲽��s
        state = (np.dot(self.U, input_array) +
                 np.dot(self.W, self.state_list[-1]))#ǰ����㹫ʽʵ��
        element_wise_op(state, self.activator.forward)#����������������
        self.state_list.append(state) #�洢�µ�ʱ�䲽״̬

	#���ഫ�����������
	#sensitivity_array, activator����������������������������������
    def backward(self, sensitivity_array, activator):
        '''
        ʵ��BPTT�㷨
        '''
        #����Ȩ�ص�������
        self.calc_delta(sensitivity_array, activator)
        #Ȩ���ݶȵļ���
        self.calc_gradient()    

    def update(self):
        '''
        �����ݶ��½�������Ȩ��
        '''
        self.W -= self.learning_rate * self.gradient#��Ȩ��=ԭȨ��-ѧϰ��*Ȩ���ݶ�
        
    #����Ȩ�ص�������
    def calc_delta(self, sensitivity_array, activator):
        self.delta_list = []  # �����������ʱ�̵������
        for i in range(self.times):
            self.delta_list.append(np.zeros(
                (self.state_width, 1)))
        self.delta_list.append(sensitivity_array)
        # ��������ÿ��ʱ�̵������
        for k in range(self.times - 1, 0, -1):
            self.calc_delta_k(k, activator)
            
    def calc_delta_k(self, k, activator):
        '''
        ����k+1ʱ�̵�delta����kʱ�̵�delta
        '''
        state = self.state_list[k+1].copy()
        element_wise_op(self.state_list[k+1],
                    activator.backward)
        self.delta_list[k] = np.dot(
            np.dot(self.delta_list[k+1].T, self.W),
            np.diag(state[:,0])).T
            
    def calc_gradient(self):
        self.gradient_list = [] # �������ʱ�̵�Ȩ���ݶ�
        for t in range(self.times + 1):
            self.gradient_list.append(np.zeros((self.state_width, self.state_width)))
        for t in range(self.times, 0, -1):
            self.calc_gradient_t(t)
        # ʵ�ʵ��ݶ��Ǹ���ʱ���ݶ�֮��
        self.gradient = reduce(
            lambda a, b: a + b, self.gradient_list,
            self.gradient_list[0]) # [0]����ʼ��Ϊ0��û�б��޸Ĺ�
            
    def calc_gradient_t(self, t):
        '''
        ����ÿ��ʱ��tȨ�ص��ݶ�
        '''
        gradient = np.dot(self.delta_list[t],
            self.state_list[t-1].T)
        self.gradient_list[t] = gradient
        
    def reset_state(self):
        self.times = 0       # ��ǰʱ�̳�ʼ��Ϊt0
        self.state_list = [] # �������ʱ�̵�state
        self.state_list.append(np.zeros(
            (self.state_width, 1)))      # ��ʼ��s0
            
def data_set():
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d
    
def gradient_check():
    '''
    �ݶȼ��
    '''
    # ���һ��������ȡ���нڵ������֮��
    error_function = lambda o: o.sum()
    
    rl = RecurrentLayer(3, 2, IdentityActivator(), 1e-3)

    # ����forwardֵ
    x, d = data_set()
    rl.forward(x[0])
    rl.forward(x[1])
    
    # ��ȡsensitivity map
    sensitivity_array = np.ones(rl.state_list[-1].shape,
                                dtype=np.float64)
    # �����ݶ�
    rl.backward(sensitivity_array, IdentityActivator())
    
    # ����ݶ�
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
