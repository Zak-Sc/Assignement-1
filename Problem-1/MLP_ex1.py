# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.


"""
import numpy as np
from math import exp

class NN(object):
        #we somewhat keept the class structure but did not use the mode, datapath and modelpath
    def __init__(self,hidden_dims=(500,500),n_hidden=2,mode='train',datapath=None,model_path=None):   
        #all weight matrices have a biais column to the right             
        self.W1 = np.zeros(shape=(hidden_dims[0],784+1))
        self.W2 = np.zeros(shape=(hidden_dims[1],hidden_dims[0]+1))
        self.W3 = np.zeros(shape=(10,hidden_dims[1]+1))
        self.y = []
        self.a1 = []
        self.a2 = []
        self.a3 = []
        self.grad_W1 = []
        self.grad_W2 = []
        self.grad_W3 = []
        
       #the initialization of weights. We keep the biais columns at zero
    def initialize_weights(self,hidden_dims, init_tech):
        # " init zeros "
        if init_tech == 1:
            self.W1=np.zeros(shape=(hidden_dims[0],784+1))
            self.W2=np.zeros(shape=(hidden_dims[1],hidden_dims[0]+1))
            self.W3=np.zeros(shape=(10,hidden_dims[1]+1))
        #" normal init"
        if init_tech == 2:  
            self.W1[:hidden_dims[0],:-1]=np.random.normal(0,1,(hidden_dims[0],784))
            self.W2[:hidden_dims[1],:-1]=np.random.normal(0,1,(hidden_dims[1],hidden_dims[0]))
            self.W3[:10,:-1]=np.random.normal(0,1,(10,hidden_dims[1]))

        #"Glorot"
        if init_tech == 3:
            self.W1[:hidden_dims[0],:-1]=np.random.uniform(-np.sqrt(6/(hidden_dims[0]+784)),np.sqrt(6/(hidden_dims[0]+784)),(hidden_dims[0],784))
            self.W2[:hidden_dims[1],:-1]=np.random.uniform(-np.sqrt(6/(hidden_dims[0]+hidden_dims[1])),np.sqrt(6/(hidden_dims[0]+hidden_dims[1])),(hidden_dims[1],hidden_dims[0]))
            self.W3[:10,:-1]=np.random.uniform(-np.sqrt(6/(hidden_dims[1]+10)),np.sqrt(6/(hidden_dims[1]+10)),(10,hidden_dims[1]))
        
    def forward(self,input,type_activation):
       #we augment the input to handle the biais
        X = np.ones(shape=(input.shape[0],input.shape[1]+1))
        X[:,:-1] = input
        
        self.a1 = np.mat(self.W1)*np.mat(X).T
       
        temph1 = self.activation(self.a1,type_activation)
        self.h1 = np.ones(shape=(temph1.shape[0]+1,temph1.shape[1]))
        self.h1[:-1,] = temph1                       
        
        self.a2 = np.mat(self.W2)*np.mat(self.h1)
        
        temph2 = self.activation(self.a2,type_activation)
        self.h2 = np.ones(shape=(temph2.shape[0]+1,temph2.shape[1]))
        self.h2[:-1,] = temph2
        
        self.a3 = np.mat(self.W3)*np.mat(self.h2)
        
        self.y = self.softmax(self.a3)
       
        return X
    
    def activation(self,input,type):
        #"tanh"
        if type==1:
            return np.tanh(input)
        #" sigmoid"
        if type==2:
            return 1.0/(1.0+np.exp(-input))
        #"relu"
        if type==3:
            return np.maximum(input,0)
    def der_activ(self,input,type):
        #"tanh"
        if type==1:
            return np.ones((input.shape[0],1)) - np.power(np.tanh(input),2)
        #"sigmoid"
        if type==2: 
            f = 1.0/(1.0+np.exp(-input))
            return np.array(f)*np.array(1.0-f)
        #"relu"
        if type==3:
            input[input<=0] = 0
            input[input>0] = 1.0
            return input
        
    #prediction
    # cross entropy and performance
    def loss(self,dataset,type_activation):
        output = []
        test = dataset[:,:-1]
        target = dataset[:,-1]
        l = 0
        self.forward(test,type_activation)
        for i in range(len(test)):
            l -= np.log(self.y[int(target[i]),i]+1/100000)
            #print (np.log(self.y[int(target[i]),i]))
        output=np.argmax(self.y,axis=0).flatten()
        performance = np.mean(target==output)
    
        return 100*performance, l/len(test)
    
    # finite difference derivative
    # we calulate the effect of modifying the first row of W2 on the prediction of train[0,:]
    def derL(self,lenght,epsilon,train,init_activ):
    
        temp = 1*self.W2
        X = np.divide(train[0,:-1],255.0)
        label = train[0,-1] * np.ones(1)
        prob=-1
        lp=[]
        lm=[]
        finit_diff=[]
        # we calculate L(theta + epsilon) using cross entropy
        for i in range(lenght):
            self.W2[0,i] = self.W2[0,i] + epsilon
            self.forward(np.reshape(X,(1,len(X))),init_activ)
            prob = self.y.item(int(label[0]))
            lp.append(-np.log(prob))
            self.W2 = 1*temp
            
        # we calculate L(theta - epsilon) using cross entropy
        for i in range(lenght):
            self.W2[0,i] = self.W2[0,i] - epsilon
            self.forward(np.reshape(X,(1,len(X))),init_activ)
            prob = self.y.item(int(label[0]))
            lm.append(-np.log(prob))
            self.W2 = 1*temp
			
		# then the finite difference derivative - the true gradient
        for i in range(lenght):
            finit_diff.append(((lp[i]-lm[i])/(2*epsilon)-gw2[0,i]) )
        
        return finit_diff

    # 
    def softmax(self,x):
        x=x.astype(float)
        if x.ndim==1:
            tempo=np.sum(np.exp(x)-np.max(x))
            return np.exp(x)/tempo
        elif x.ndim==2:
            tempo=np.zeros_like(x)
            M,N=x.shape
            for i in range(N):
                S=np.sum(np.exp(x[:,i]-np.max(x[:,i])))
                tempo[:,i]=np.exp(x[:,i]-np.max(x[:,i]))/S
            return tempo
   
    def backward(self,X,label,type_activation):
        
        #gradient with respect to a_3
        grad_a3 = self.y
        for i in range(self.y.shape[1]):
            grad_a3[int(label[i]),i] = self.y[int(label[i]),i]-1
        
        #gradient with respect to W3
        grad_W3 = np.mat(grad_a3)*self.h2.T
        
        #gradient with respect to h2
        W3T = np.transpose(np.mat(self.W3[:,:-1]))
       
       # grad_h2 = np.dot(np.array(W3T,grad_a_kp1))
        grad_h2 = np.mat(W3T)*np.mat(grad_a3)
        
        #gradient with respect to a2
        grad_a2 = (np.array(grad_h2) * np.array((self.der_activ(self.a2,type_activation))))
        
        #gradient with respect to W2
        grad_W2 = np.mat(grad_a2)*self.h1.T
        
        #gradient with respect to h1
        W2T = np.transpose(np.mat(self.W2[:,:-1]))
        grad_h1 = np.mat(W2T)*np.mat(grad_a2)
        
        #gradient with respect to a1
        grad_a1 = (np.array(grad_h1) * np.array((self.der_activ(self.a1,type_activation))))
        
        #gradient with respect to W1
        grad_W1 = np.mat(grad_a1)*X
        
        return grad_W3, grad_W2, grad_W1
        
        # uses the gradients, the step size (mu) and the regularization (lambda)
    def update(self,lambd,mu,grad_W3, grad_W2, grad_W1):

        self.W1 = np.mat(self.W1) - mu*np.mat(grad_W1) - mu*2*lambd*np.mat(self.W1)
        self.W2 = np.mat(self.W2) - mu*np.mat(grad_W2) - mu*2*lambd*np.mat(self.W2)
        self.W3 = np.mat(self.W3) - mu*np.mat(grad_W3) - mu*2*lambd*np.mat(self.W3)
   
    def train(self,train_sample,hidden_dims,init_tech,init_activ,batchsize,lambd,mu,n_epoch):
        #to make result reproducible
        np.random.seed(234)
        sample_size = train_sample.shape[0]
        # initialization of weights
        self.initialize_weights(hidden_dims,init_tech)
        i=0
        perfo1=0
        perfo2= 0
        p_data=[]
        n_iter=0
        max_iter=sample_size*n_epoch
        
        while n_iter<max_iter:
            train_ex = train_sample[i:i+batchsize,:-1]      #collect the training samples
            norm_train_ex = np.divide(train_ex,255.0)    #divide the data by 255
            label = train_sample[i:i+batchsize,-1]          #collect labels
            X = self.forward(norm_train_ex,init_activ)      #forward
            [grad_W3, grad_W2, grad_W1]=self.backward(X,label,init_activ)   #backprop
            # use gradients to update weights
            # lambda is divided by the sample size
            self.update(lambd/sample_size,mu, np.mat(grad_W3),  np.mat(grad_W2), np.mat(grad_W1))
            
            i=(i+batchsize)%sample_size
            n_iter +=batchsize
            # after each epoch compute performane on train and valid
            if i%50000 == 0:
              [perfo1,loss_train] = self.loss(train_sample,init_activ)
              [perfo2, loss_valid] = self.loss(valid,init_activ)
              p_data.append((perfo1,perfo2))
              print("acc train:" + str(perfo1) +",acc_valid:" + str(perfo2) +",loss_train:"+str(loss_train)+", loss_valid:"+str(loss_valid))
        
        return grad_W3, grad_W2, grad_W1,p_data   
    
    #   I dont always test my model but when I do, I do it on the validation set
    def test(self):
        pass

    # run mnist.py before to import the data
if __name__ == '__main__':
       
       RN = NN(hidden_dims=(900,300))
       
       valid = train[50001:60000,:]
       #init tech [1=zero,2=normal,3=glorot]    init_activ [1=tanh,2=sigmoid,3=relu], 
       #batchsize,lambda,n_epoch
       [gw3,gw2,gw1,p_data]=RN.train(train[:100,:],(900,300),3,3,20,0.05,0.01,10)
 
       # derivatives for 15 points
       deriv10=RN.derL(15,1/10,train,1)
       deriv100=RN.derL(15,1/100,train,1)
       deriv1000=RN.derL(15,1/1000,train,1)
       deriv10000=RN.derL(15,1/10000,train,1)
       deriv100000=RN.derL(15,1/100000,train,1)
   