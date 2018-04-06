#include "iostream"
#include "stdio.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "time.h"
#include "vector"
#include "assert.h"

using namespace std;
   
#define innode  2       //输入结点数，将输入2个加数
#define hidenode  56    //隐藏结点数，存储“携带位”  越大预测越准，时间越长 
#define outnode  1      //输出结点数，将输出一个预测数字
#define alpha  0.01      //学习速率  改小防止反弹，但时间延长 
#define lenoftrain  20   //训练数据长度 

#define randval(high) ( (double)rand() / RAND_MAX * high )
#define uniform_plus_minus_one ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 ) 

int init_data[lenoftrain]=
{8,2,1,8,2,0,5,0,0,1,0,3,4,6,7,0,6,1,2,3
 };
int train_dim=lenoftrain-2;//训练时的预测数据长度 

//Identity
double identity(double x) 
{
    return x ;
}

//Relu
double relu(double x) 
{
    return fmax(0,x) ;
}

//sigmoid
double sigmoid(double x) 
{
	
    return 1.0 / (1.0 + exp(-x));
}

//sigmoid的导数_re
double dsigmoid_re(double y)
{
	y=sigmoid(y); 
    return y * (1.0 - y);  
}    
 
//sigmoid的导数
double dsigmoid(double y)
{
	//y=sigmoid(y); 
    return y * (1.0 - y);  
}                 
        
//tanh的导数，y为tanh值
double dtanh(double y)
{
	y=tanh(y);
    return 1.0 - y * y;  
}

void winit(double w[], int n) //权值初始化
{
    for(int i=0; i<n; i++)
        w[i] = uniform_plus_minus_one;  //-1~1均匀随机分布
}

class RNN
{
	public:
	    RNN();//构造函数 
	    virtual ~RNN();//析构函数 
	    void train();
	    void data_set(int* initdata,int* data0,int* data1,int* data2,int len);
	    void show_result(int timesoftrain,int* predict,int* data2);
	    void forword(int timesoftrain,int* data0,int* data1,int* data2,int* predict);
	    void backword(int* data0,int* data1);
	    
	public:
	    double W_I[innode][hidenode];     //连接输入与隐含层单元中   输入门   的权值矩阵
	    double U_I[hidenode][hidenode];   //连接上一隐层输出与本隐含层单元中输入门的权值矩阵
	    double W_F[innode][hidenode];     //连接输入与隐含层单元中   遗忘门   的权值矩阵
	    double U_F[hidenode][hidenode];   //连接上一隐含层与本隐含层单元中  遗忘门   的权值矩阵
	    double W_O[innode][hidenode];     //连接输入与隐含层单元中输出门的权值矩阵
	    double U_O[hidenode][hidenode];   //连接上一隐含层与现在时刻的隐含层的权值矩阵
	    double W_G[innode][hidenode];     //用于产生新记忆的权值矩阵
	    double U_G[hidenode][hidenode];   //用于产生新记忆的权值矩阵
	    double W_out[hidenode][outnode];  //连接隐层与输出层的权值矩阵
	
	    double *x;             //layer 0 输出值，由输入向量直接设定
	    double *y;             //layer 2 输出值
	    double error;
	    int    predict[lenoftrain];       //预测的结果 
	    
	    vector<double*> I_vector;      //输入门
   	 	vector<double*> F_vector;      //遗忘门
    	vector<double*> O_vector;      //输出门
    	vector<double*> G_vector;      //新记忆
    	vector<double*> S_vector;      //状态值
    	vector<double*> h_vector;      //输出值
    	vector<double>  y_delta;       //保存误差关于输出层的偏导
};

void RNN::data_set(int* initdata,int* data0,int* data1,int* data2,int len)
{
	for(int i=0;i<len;i++)
	{
		data0[i]=initdata[i];
		//cout<<data0[i]<<endl;
	}
	for(int i=0;i<lenoftrain-1;i++)
	{
		data1[i]=initdata[i+1];
		//cout<<data1[i]<<endl;
	}
    for(int i=0;i<lenoftrain-2;i++)
	{
		data2[i]=initdata[i+2];
		//cout<<data2[i]<<endl;
	}
}

RNN::RNN()
{
    x = new double[innode]; //给输入矩阵申请的内存 
    y = new double[outnode];
    winit((double*)W_I, innode * hidenode);
    winit((double*)U_I, hidenode * hidenode);
    winit((double*)W_F, innode * hidenode);	
    winit((double*)U_F, hidenode * hidenode);
    winit((double*)W_O, innode * hidenode);
    winit((double*)U_O, hidenode * hidenode);
    winit((double*)W_G, innode * hidenode);
    winit((double*)U_G, hidenode * hidenode);
    winit((double*)W_out, hidenode * outnode);
}

RNN::~RNN()
{
    delete x;
    delete y;
}

void RNN::train()
{
	int timesoftrain;
	int time_data0[lenoftrain];//单种虚拟机在时间轴下的台数信息    
	int time_data1[lenoftrain];//单种虚拟机在时间轴下的台数信息       错位时间 
	int time_data2[lenoftrain];//单种虚拟机在时间轴下的台数信息       错位时间 
    
    data_set(init_data,time_data0,time_data1,time_data2,lenoftrain);//剪裁成时间错位的数组 
    
    for(timesoftrain=0; timesoftrain<30000; timesoftrain++) //第timesoftrain+1次训练开始 
    {
    	error = 0.0;
		memset(predict, 0, sizeof(predict));
        
        //在0时刻是没有之前的隐含层的，所以初始化一个全为0的
        double *S = new double[hidenode];     //状态值  Ct候选值 
        double *h = new double[hidenode];     //隐藏层输出值 h
        for(int i=0; i<hidenode; i++)  
        {
            S[i] = 0;
            h[i] = 0;
        }
        S_vector.push_back(S);
        h_vector.push_back(h); 
        
        forword(timesoftrain,time_data0,time_data1,time_data2,predict); //前馈传播 	
        
        backword(time_data0,time_data1);//反馈 
		
		show_result(timesoftrain,predict,time_data2);
		
		for(int i=0; i<I_vector.size(); i++)
            delete I_vector[i];
        for(int i=0; i<F_vector.size(); i++)
            delete F_vector[i];
        for(int i=0; i<O_vector.size(); i++)
            delete O_vector[i];
        for(int i=0; i<G_vector.size(); i++)
            delete G_vector[i];
        for(int i=0; i<S_vector.size(); i++)
            delete S_vector[i];
        for(int i=0; i<h_vector.size(); i++)
            delete h_vector[i];

        I_vector.clear();
        F_vector.clear();
        O_vector.clear();
        G_vector.clear();
        S_vector.clear();
        h_vector.clear();
        y_delta.clear();
    } 
} 

void RNN::backword(int* data0,int* data1)
{
	//隐含层偏差，通过当前之后一个时间点的隐含层误差和当前输出层的误差计算
    double h_delta[hidenode];  
    double *O_delta = new double[hidenode];
    double *I_delta = new double[hidenode];
    double *F_delta = new double[hidenode];
    double *G_delta = new double[hidenode];
    double *state_delta = new double[hidenode];
    //当前时间之后的一个隐藏层误差
    double *O_future_delta = new double[hidenode]; 
    double *I_future_delta = new double[hidenode];
    double *F_future_delta = new double[hidenode];
    double *G_future_delta = new double[hidenode];
    double *state_future_delta = new double[hidenode];
    double *forget_gate_future = new double[hidenode];
	for(int j=0; j<hidenode; j++)
    {
        O_future_delta[j] = 0;
        I_future_delta[j] = 0;
        F_future_delta[j] = 0;
        G_future_delta[j] = 0;
        state_future_delta[j] = 0;
        forget_gate_future[j] = 0;
    }
    
    for(int p=train_dim-1; p>=0 ; p--)
    {
		x[0] = data0[p];
        x[1] = data1[p];
        //cout<<data0[p]<<endl;
        
        //当前隐藏层
        double *in_gate = I_vector[p];     //输入门
        double *out_gate = O_vector[p];    //输出门
        double *forget_gate = F_vector[p]; //遗忘门
        double *g_gate = G_vector[p];      //新记忆
        double *state = S_vector[p+1];     //状态值
        double *h = h_vector[p+1];         //隐层输出值
        
        //前一个隐藏层
        double *h_pre = h_vector[p];   
        double *state_pre = S_vector[p];
        
        for(int k=0; k<outnode; k++)  //对于网络中每个输出单元，更新权值
        {
            //更新隐含层和输出层之间的连接权
            for(int j=0; j<hidenode; j++)
                W_out[j][k] += alpha * y_delta[p] * h[j];  //输出层权重矩阵更新 
        }
        
        //对于网络中每个隐藏单元，计算误差项，并更新权值
        for(int j=0; j<hidenode; j++) 
        {
			h_delta[j] = 0.0;
            for(int k=0; k<outnode; k++)
            {
                h_delta[j] += y_delta[p] * W_out[j][k];
                //cout<<y_delta[p]<<endl;
            }
            
            for(int k=0; k<hidenode; k++)
            {
                h_delta[j] += I_future_delta[k] * U_I[j][k];
                h_delta[j] += F_future_delta[k] * U_F[j][k];
                h_delta[j] += O_future_delta[k] * U_O[j][k];
                h_delta[j] += G_future_delta[k] * U_G[j][k];
            }
            
            O_delta[j] = 0.0;
            I_delta[j] = 0.0;
            F_delta[j] = 0.0;
            G_delta[j] = 0.0;
            state_delta[j] = 0.0;
            
            //隐含层的校正误差
            O_delta[j] = h_delta[j] * tanh(state[j]) * dsigmoid(out_gate[j]);
            state_delta[j] = h_delta[j] * out_gate[j] * dtanh(state[j]) +state_future_delta[j] * forget_gate_future[j];
            F_delta[j] = state_delta[j] * state_pre[j] * dsigmoid(forget_gate[j]);
            I_delta[j] = state_delta[j] * g_gate[j] * dsigmoid_re(in_gate[j]);
            G_delta[j] = state_delta[j] * in_gate[j] * dsigmoid_re(g_gate[j]);            
            
            //更新前一个隐含层和现在隐含层之间的权值
            for(int k=0; k<hidenode; k++)
            {
                U_I[k][j] += alpha * I_delta[j] * h_pre[k];
                U_F[k][j] += alpha * F_delta[j] * h_pre[k];
                U_O[k][j] += alpha * O_delta[j] * h_pre[k];
                U_G[k][j] += alpha * G_delta[j] * h_pre[k];
            }   
			
            //更新输入层和隐含层之间的连接权
            for(int k=0; k<innode; k++)
            {
                W_I[k][j] += alpha * I_delta[j] * x[k];
                W_F[k][j] += alpha * F_delta[j] * x[k];
                W_O[k][j] += alpha * O_delta[j] * x[k];
                W_G[k][j] += alpha * G_delta[j] * x[k];
            }			          
		}
		
		//cout<<"I_delta: ";
		//for(int j=0; j<hidenode; j++) 
        //{
			//cout<<I_delta[j];
			//cout<<" ";
		//}
		//cout<<endl;
		
		//cout<<"F_delta: ";
		//for(int j=0; j<hidenode; j++) 
        //{
			//cout<<F_delta[j];
			//cout<<" ";
			
		//}
		//cout<<endl;
		
		//cout<<"O_delta: ";
		//for(int j=0; j<hidenode; j++) 
        //{
			//cout<<O_delta[j];
			//cout<<" ";
		
		//}
		//cout<<endl;
		
		//cout<<"G_delta: ";
		//for(int j=0; j<hidenode; j++) 
        //{
			//cout<<G_delta[j];
			//cout<<" ";
		//}
		//cout<<endl;
		
		if(p == train_dim-1)//清除上一轮的数据 
        {
	        delete  O_future_delta;
	        delete  F_future_delta;
	        delete  I_future_delta;
	        delete  G_future_delta;
	        delete  state_future_delta;
	        delete  forget_gate_future;
        }
        
        O_future_delta = O_delta;
        F_future_delta = F_delta;
        I_future_delta = I_delta;
        G_future_delta = G_delta;
        state_future_delta = state_delta;
        forget_gate_future = forget_gate;
        
        //cout<<"h_pre: ";
	    //for(int k=0; k<hidenode; k++)
	    //{
	        //cout<<h_pre[k];
	        //cout<<" ";
	    //}
	    //cout<<endl;
		
	}
	
	delete  O_future_delta;
    delete  F_future_delta;
    delete  I_future_delta;
    delete  G_future_delta;
    delete  state_future_delta;

    ///cout<<"U_I: ";
    //for(int i=0;i<hidenode;i++)
    //{
    	//for(int j=0;j<hidenode;j++)
    	//{
			//cout<<U_I[i][j];
			//cout<<" ";
		//}		
	//}
	//cout<<endl;  
	   
    //cout<<"U_F: ";
    //for(int i=0;i<hidenode;i++)
    //{
    	//for(int j=0;j<hidenode;j++)
    	//{
			//cout<<U_F[i][j];
			//cout<<" ";
		//}		
	//}
	//cout<<endl;   
  
    //cout<<"U_O: ";
    //for(int i=0;i<hidenode;i++)
    //{
    	//for(int j=0;j<hidenode;j++)
    	//{
			//cout<<U_O[i][j];
			//cout<<" ";
		//}		
	//}
	//cout<<endl;   
  
    //cout<<"U_G: ";
    //for(int i=0;i<hidenode;i++)
    //{
    	//for(int j=0;j<hidenode;j++)
    	//{
			//cout<<U_G[i][j];
			//cout<<" ";
		//}		
	//}
	//cout<<endl;     
    
    //cout<<"W_I: ";
    //for(int i=0;i<innode;i++)
    //{
    	//for(int j=0;j<hidenode;j++)
    	//{
			//cout<<W_I[i][j];
			//cout<<" ";
		//}		
	//}
	//cout<<endl;    
    
    //cout<<"W_F: ";
    //for(int i=0;i<innode;i++)
    //{
    	//for(int j=0;j<hidenode;j++)
    	//{
			//cout<<W_F[i][j];
			//cout<<" ";
		//}		
	//}
	//cout<<endl;    
    
    //cout<<"W_O: ";
    //for(int i=0;i<innode;i++)
    //{
    	//for(int j=0;j<hidenode;j++)
    	//{
			//cout<<W_O[i][j];
			//cout<<" ";
		//}		
	//}
	//cout<<endl;  
    
    //cout<<"W_G: ";
    //for(int i=0;i<innode;i++)
    //{
    	//for(int j=0;j<hidenode;j++)
    	//{
			//cout<<W_G[i][j];
			//cout<<" ";
		//}		
	//}
	//cout<<endl;    
    
    //cout<<"W_out: ";
    //for(int i=0;i<hidenode;i++)
    //{
    	//for(int j=0;j<outnode;j++)
    	//{
			//cout<<W_out[i][j];
			//cout<<" ";
		//}		
	//}
	//cout<<endl;
}

void RNN::show_result(int timesoftrain,int* predict,int* data2) 
{
	if(timesoftrain % 100 == 0)
    {
		printf("第%d00次训练结束\n",timesoftrain/100);
			cout << "总误差：" << error << endl;
	
		cout << "预测结果：" ;
		for(int k=0; k<lenoftrain-2; k++)
		{
	   		cout << predict[k];
	   		cout << " " ;	
		}  
    	cout << endl;
    
    	cout << "真实值：  " ;
    	for(int k=0; k<lenoftrain-2; k++)
    	{
	   		cout << data2[k];
	   		cout << " " ;	
		}
		cout << "\n" ;
    	cout << endl;
	}
}

void RNN::forword(int timesoftrain,int* data0,int* data1,int* data2,int* predict)
{
	
	for(int p=0; p<train_dim; p++)           //循环遍历所有数，从前往后 
    {
    	
		x[0] = data0[p];
        x[1] = data1[p];//输入矩阵采集 前98步训练数据到位 
        int answer = data2[p]; //当前步答案 
        //cout<<x[0]<<endl;
        
        double *in_gate = new double[hidenode];     //输入门
    	double *out_gate = new double[hidenode];    //输出门
        double *forget_gate = new double[hidenode]; //遗忘门
        double *g_gate = new double[hidenode];      //新记忆 ~Ct
    	double *state = new double[hidenode];       //状态值 也即候选值Ct 
        double *h = new double[hidenode];           //隐层值  
        
        for(int j=0; j<hidenode; j++)
        { 
			//输入层转播到隐层的前馈传播 
			double inGate = 0.0;
            double outGate = 0.0;
            double forgetGate = 0.0;
            double gGate = 0.0;
            double s = 0.0;
            
            //in_gate[j] = sigmoid(inGate)
            //inGate=Wi[h_t-1,x_t]+bi=
            //  W_I*x+U_I*h_t-1+bi
                
            //forget_gate[j] = sigmoid(forgetGate);
            //forgetGate=Wf[h_t-1,x_t]+bf=
            //  W_F*x+U_F*h_t-1+bf
                
            //out_gate[j] = sigmoid(outGate);
            //outGate=Wo[h_t-1,x_t]+bo=
            //  W_O*x+U_O*h_t-1+bo
                
            //g_gate[j] = sigmoid(gGate);//新记忆 
            //gGate=Wg[h_t-1,x_t]+bg=
            //  W_G*x+U_G*h_t-1+bg   
			for(int m=0; m<innode; m++) 
            {
            	//cout<<W_I[m][j]<<endl;
                inGate += x[m] * W_I[m][j]; //输入门 
                outGate += x[m] * W_O[m][j]; //输出门 
                forgetGate += x[m] * W_F[m][j]; //遗忘门 
                gGate += x[m] * W_G[m][j]; //新记忆 ~Ct
            }  
			double *h_pre = h_vector.back();//h_t-1 
            double *state_pre = S_vector.back(); //Ct    
			for(int m=0; m<hidenode; m++)
            {
                inGate += h_pre[m] * U_I[m][j];
                outGate += h_pre[m] * U_O[m][j];
                forgetGate += h_pre[m] * U_F[m][j];
                gGate += h_pre[m] * U_G[m][j];
            }  

            //偏置层 
            inGate += 1;
            outGate += 1;
            forgetGate += 1;
            gGate += 1;
            in_gate[j] = sigmoid(inGate);   
            out_gate[j] = sigmoid(outGate);
            forget_gate[j] = sigmoid(forgetGate);
            g_gate[j] = tanh(gGate);
            
            double s_pre = state_pre[j]; //上一个候选值 
            // state[j] 候选值C 
            //Ct=ft*Ct-1+it*(~Ct)
            state[j] = forget_gate[j] * s_pre + in_gate[j]* g_gate[j] ;
            
            //h 隐藏层矩阵 
            //ht=ot*tanh(Ct)
            h[j] = out_gate[j] * tanh(state[j]);
            
        }
		
		for(int k=0; k<outnode; k++)
        {
            //隐藏层传播到输出层的前馈传播 
            double out = 0.0;
            for(int j=0; j<hidenode; j++)
                out += h[j] * W_out[j][k];
			//output=sigmoid(W_out*h_t+b_out)  
			//y[k] = sigmoid(out); //输出层各单元输出   
			y[k] = identity(out);
        }
        predict[p] = (int) round(y[0]);  //记录预测值 预测的数组大小就是输入数组那么大 
        
        //保存隐藏层 此时只进行了一个时间步的前馈计算 
        I_vector.push_back(in_gate);
        F_vector.push_back(forget_gate);
        O_vector.push_back(out_gate);
        S_vector.push_back(state);
        G_vector.push_back(g_gate);
        h_vector.push_back(h);
		 
		//保存标准误差关于输出层的偏导
        y_delta.push_back( (answer - y[0]) * dsigmoid_re(y[0]) );
        
        //cout<<"y_delta: ";
        //cout<< (answer - y[0]) * dsigmoid(y[0]);
        //cout<<" ";
		//cout<<endl;
        
        //cout<<"erro_ones";
        //cout<< (answer - y[0]);
        //cout<<" ";
		//cout<<endl;
    	error += fabs(answer - y[0]);   //误差（累计train_dim次，重新开始时会清0）
	} 
}

int main()
{
	srand(time(NULL));
	RNN rnn;
    rnn.train(); 
    
    return 0;
}
