#include "iostream"
#include "stdio.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "time.h"
#include "vector"
#include "assert.h"

using namespace std;
   
#define innode  2       //����������������2������
#define hidenode  56    //���ؽ�������洢��Я��λ��  Խ��Ԥ��Խ׼��ʱ��Խ�� 
#define outnode  1      //���������������һ��Ԥ������
#define alpha  0.01      //ѧϰ����  ��С��ֹ��������ʱ���ӳ� 
#define lenoftrain  20   //ѵ�����ݳ��� 

#define randval(high) ( (double)rand() / RAND_MAX * high )
#define uniform_plus_minus_one ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 ) 

int init_data[lenoftrain]=
{8,2,1,8,2,0,5,0,0,1,0,3,4,6,7,0,6,1,2,3
 };
int train_dim=lenoftrain-2;//ѵ��ʱ��Ԥ�����ݳ��� 

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

//sigmoid�ĵ���_re
double dsigmoid_re(double y)
{
	y=sigmoid(y); 
    return y * (1.0 - y);  
}    
 
//sigmoid�ĵ���
double dsigmoid(double y)
{
	//y=sigmoid(y); 
    return y * (1.0 - y);  
}                 
        
//tanh�ĵ�����yΪtanhֵ
double dtanh(double y)
{
	y=tanh(y);
    return 1.0 - y * y;  
}

void winit(double w[], int n) //Ȩֵ��ʼ��
{
    for(int i=0; i<n; i++)
        w[i] = uniform_plus_minus_one;  //-1~1��������ֲ�
}

class RNN
{
	public:
	    RNN();//���캯�� 
	    virtual ~RNN();//�������� 
	    void train();
	    void data_set(int* initdata,int* data0,int* data1,int* data2,int len);
	    void show_result(int timesoftrain,int* predict,int* data2);
	    void forword(int timesoftrain,int* data0,int* data1,int* data2,int* predict);
	    void backword(int* data0,int* data1);
	    
	public:
	    double W_I[innode][hidenode];     //���������������㵥Ԫ��   ������   ��Ȩֵ����
	    double U_I[hidenode][hidenode];   //������һ��������뱾�����㵥Ԫ�������ŵ�Ȩֵ����
	    double W_F[innode][hidenode];     //���������������㵥Ԫ��   ������   ��Ȩֵ����
	    double U_F[hidenode][hidenode];   //������һ�������뱾�����㵥Ԫ��  ������   ��Ȩֵ����
	    double W_O[innode][hidenode];     //���������������㵥Ԫ������ŵ�Ȩֵ����
	    double U_O[hidenode][hidenode];   //������һ������������ʱ�̵��������Ȩֵ����
	    double W_G[innode][hidenode];     //���ڲ����¼����Ȩֵ����
	    double U_G[hidenode][hidenode];   //���ڲ����¼����Ȩֵ����
	    double W_out[hidenode][outnode];  //����������������Ȩֵ����
	
	    double *x;             //layer 0 ���ֵ������������ֱ���趨
	    double *y;             //layer 2 ���ֵ
	    double error;
	    int    predict[lenoftrain];       //Ԥ��Ľ�� 
	    
	    vector<double*> I_vector;      //������
   	 	vector<double*> F_vector;      //������
    	vector<double*> O_vector;      //�����
    	vector<double*> G_vector;      //�¼���
    	vector<double*> S_vector;      //״ֵ̬
    	vector<double*> h_vector;      //���ֵ
    	vector<double>  y_delta;       //����������������ƫ��
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
    x = new double[innode]; //���������������ڴ� 
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
	int time_data0[lenoftrain];//�����������ʱ�����µ�̨����Ϣ    
	int time_data1[lenoftrain];//�����������ʱ�����µ�̨����Ϣ       ��λʱ�� 
	int time_data2[lenoftrain];//�����������ʱ�����µ�̨����Ϣ       ��λʱ�� 
    
    data_set(init_data,time_data0,time_data1,time_data2,lenoftrain);//���ó�ʱ���λ������ 
    
    for(timesoftrain=0; timesoftrain<30000; timesoftrain++) //��timesoftrain+1��ѵ����ʼ 
    {
    	error = 0.0;
		memset(predict, 0, sizeof(predict));
        
        //��0ʱ����û��֮ǰ��������ģ����Գ�ʼ��һ��ȫΪ0��
        double *S = new double[hidenode];     //״ֵ̬  Ct��ѡֵ 
        double *h = new double[hidenode];     //���ز����ֵ h
        for(int i=0; i<hidenode; i++)  
        {
            S[i] = 0;
            h[i] = 0;
        }
        S_vector.push_back(S);
        h_vector.push_back(h); 
        
        forword(timesoftrain,time_data0,time_data1,time_data2,predict); //ǰ������ 	
        
        backword(time_data0,time_data1);//���� 
		
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
	//������ƫ�ͨ����ǰ֮��һ��ʱ�������������͵�ǰ������������
    double h_delta[hidenode];  
    double *O_delta = new double[hidenode];
    double *I_delta = new double[hidenode];
    double *F_delta = new double[hidenode];
    double *G_delta = new double[hidenode];
    double *state_delta = new double[hidenode];
    //��ǰʱ��֮���һ�����ز����
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
        
        //��ǰ���ز�
        double *in_gate = I_vector[p];     //������
        double *out_gate = O_vector[p];    //�����
        double *forget_gate = F_vector[p]; //������
        double *g_gate = G_vector[p];      //�¼���
        double *state = S_vector[p+1];     //״ֵ̬
        double *h = h_vector[p+1];         //�������ֵ
        
        //ǰһ�����ز�
        double *h_pre = h_vector[p];   
        double *state_pre = S_vector[p];
        
        for(int k=0; k<outnode; k++)  //����������ÿ�������Ԫ������Ȩֵ
        {
            //����������������֮�������Ȩ
            for(int j=0; j<hidenode; j++)
                W_out[j][k] += alpha * y_delta[p] * h[j];  //�����Ȩ�ؾ������ 
        }
        
        //����������ÿ�����ص�Ԫ����������������Ȩֵ
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
            
            //�������У�����
            O_delta[j] = h_delta[j] * tanh(state[j]) * dsigmoid(out_gate[j]);
            state_delta[j] = h_delta[j] * out_gate[j] * dtanh(state[j]) +state_future_delta[j] * forget_gate_future[j];
            F_delta[j] = state_delta[j] * state_pre[j] * dsigmoid(forget_gate[j]);
            I_delta[j] = state_delta[j] * g_gate[j] * dsigmoid_re(in_gate[j]);
            G_delta[j] = state_delta[j] * in_gate[j] * dsigmoid_re(g_gate[j]);            
            
            //����ǰһ�������������������֮���Ȩֵ
            for(int k=0; k<hidenode; k++)
            {
                U_I[k][j] += alpha * I_delta[j] * h_pre[k];
                U_F[k][j] += alpha * F_delta[j] * h_pre[k];
                U_O[k][j] += alpha * O_delta[j] * h_pre[k];
                U_G[k][j] += alpha * G_delta[j] * h_pre[k];
            }   
			
            //����������������֮�������Ȩ
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
		
		if(p == train_dim-1)//�����һ�ֵ����� 
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
		printf("��%d00��ѵ������\n",timesoftrain/100);
			cout << "����" << error << endl;
	
		cout << "Ԥ������" ;
		for(int k=0; k<lenoftrain-2; k++)
		{
	   		cout << predict[k];
	   		cout << " " ;	
		}  
    	cout << endl;
    
    	cout << "��ʵֵ��  " ;
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
	
	for(int p=0; p<train_dim; p++)           //ѭ����������������ǰ���� 
    {
    	
		x[0] = data0[p];
        x[1] = data1[p];//�������ɼ� ǰ98��ѵ�����ݵ�λ 
        int answer = data2[p]; //��ǰ���� 
        //cout<<x[0]<<endl;
        
        double *in_gate = new double[hidenode];     //������
    	double *out_gate = new double[hidenode];    //�����
        double *forget_gate = new double[hidenode]; //������
        double *g_gate = new double[hidenode];      //�¼��� ~Ct
    	double *state = new double[hidenode];       //״ֵ̬ Ҳ����ѡֵCt 
        double *h = new double[hidenode];           //����ֵ  
        
        for(int j=0; j<hidenode; j++)
        { 
			//�����ת���������ǰ������ 
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
                
            //g_gate[j] = sigmoid(gGate);//�¼��� 
            //gGate=Wg[h_t-1,x_t]+bg=
            //  W_G*x+U_G*h_t-1+bg   
			for(int m=0; m<innode; m++) 
            {
            	//cout<<W_I[m][j]<<endl;
                inGate += x[m] * W_I[m][j]; //������ 
                outGate += x[m] * W_O[m][j]; //����� 
                forgetGate += x[m] * W_F[m][j]; //������ 
                gGate += x[m] * W_G[m][j]; //�¼��� ~Ct
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

            //ƫ�ò� 
            inGate += 1;
            outGate += 1;
            forgetGate += 1;
            gGate += 1;
            in_gate[j] = sigmoid(inGate);   
            out_gate[j] = sigmoid(outGate);
            forget_gate[j] = sigmoid(forgetGate);
            g_gate[j] = tanh(gGate);
            
            double s_pre = state_pre[j]; //��һ����ѡֵ 
            // state[j] ��ѡֵC 
            //Ct=ft*Ct-1+it*(~Ct)
            state[j] = forget_gate[j] * s_pre + in_gate[j]* g_gate[j] ;
            
            //h ���ز���� 
            //ht=ot*tanh(Ct)
            h[j] = out_gate[j] * tanh(state[j]);
            
        }
		
		for(int k=0; k<outnode; k++)
        {
            //���ز㴫����������ǰ������ 
            double out = 0.0;
            for(int j=0; j<hidenode; j++)
                out += h[j] * W_out[j][k];
			//output=sigmoid(W_out*h_t+b_out)  
			//y[k] = sigmoid(out); //��������Ԫ���   
			y[k] = identity(out);
        }
        predict[p] = (int) round(y[0]);  //��¼Ԥ��ֵ Ԥ��������С��������������ô�� 
        
        //�������ز� ��ʱֻ������һ��ʱ�䲽��ǰ������ 
        I_vector.push_back(in_gate);
        F_vector.push_back(forget_gate);
        O_vector.push_back(out_gate);
        S_vector.push_back(state);
        G_vector.push_back(g_gate);
        h_vector.push_back(h);
		 
		//�����׼������������ƫ��
        y_delta.push_back( (answer - y[0]) * dsigmoid_re(y[0]) );
        
        //cout<<"y_delta: ";
        //cout<< (answer - y[0]) * dsigmoid(y[0]);
        //cout<<" ";
		//cout<<endl;
        
        //cout<<"erro_ones";
        //cout<< (answer - y[0]);
        //cout<<" ";
		//cout<<endl;
    	error += fabs(answer - y[0]);   //���ۼ�train_dim�Σ����¿�ʼʱ����0��
	} 
}

int main()
{
	srand(time(NULL));
	RNN rnn;
    rnn.train(); 
    
    return 0;
}
