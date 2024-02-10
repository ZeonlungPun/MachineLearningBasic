#softmax regression dealing with multiple classification
import numpy as np
import pandas as pd
data=pd.read_excel('wine1.xlsx',engine='openpyxl')
data=data.iloc[:,1::]
x=data.iloc[:,0:13]
(m,n)=x.shape
y=data.iloc[:,-4::]
y=np.array(y)
y=y.reshape(m,4)
x=np.array(x)
x=x.reshape(m,n)
x=(x-np.min(x))/(np.max(x)-np.min(x))#归一化
random_number=np.random.permutation(m)#产生随机数序列
random_number=random_number[:,np.newaxis]#升维操作，以使其和x，y维度对应
random_number.reshape(m,1)
random_num=round(m*0.3)#test set number
z11=random_number[1:random_num,:]#测试集所对应的样本序号
z12=random_number[(random_num+1)::,:]#训练集所对应的样本序号
xtest,ytest=x[z11,:],y[z11,-1]
xtrain,ytrain=x[z12,:],y[z12,-1]
xtrain=np.squeeze(xtrain,axis=1)
xtest=np.squeeze(xtest,axis=1)
t1=y[z11,-4:-1]#test set one-hot code
t2=y[z12,-4:-1]#train set one-hot code
t1=np.squeeze(t1,axis=1)
t2=np.squeeze(t2,axis=1)
w=np.ones((3,xtrain.shape[1]))#initialisation,weight
w0=np.ones((3,1))
epochs=20000#iteration number
lr=0.05#learning rate
sigma1=w
sigma2=w0
alpha=0.1#weight of old gradient
#RMS Prop
for i in range(epochs+1):
    w1,w2,w3=w[0],w[1],w[2]
    w01,w02,w03=w0[0],w0[1],w0[2]#注意索引从0开始
    seg1=np.exp(np.dot(xtrain,w1.T)+w01)
    seg2=np.exp(np.dot(xtrain,w2.T)+w02)
    seg3=np.exp(np.dot(xtrain,w3.T)+w03)
    total=seg1+seg2+seg3
    z1=seg1/total#class1
    z2=seg2/total#class2
    z3=seg3/total#class3
    t21,t22,t23=t2[:,0],t2[:,1],t2[:,2]
    deltaw1=(sum((z1.reshape(xtrain.shape[0],1)-t21.reshape((xtrain.shape[0],1)))*xtrain)).reshape((1,-1))
    deltaw2=(sum((z2.reshape(xtrain.shape[0],1)-t22.reshape((xtrain.shape[0],1)))*xtrain)).reshape((1,-1))
    deltaw3=(sum((z3.reshape(xtrain.shape[0],1)-t23.reshape((xtrain.shape[0],1)))*xtrain)).reshape((1,-1))
    deltaw01=(sum(z1.reshape(xtrain.shape[0],1)-t21.reshape((xtrain.shape[0],1)))).reshape((1,1))
    deltaw02=(sum(z2.reshape(xtrain.shape[0],1)-t22.reshape((xtrain.shape[0],1)))).reshape((1,1))
    deltaw03=(sum(z3.reshape(xtrain.shape[0],1)-t23.reshape((xtrain.shape[0],1)))).reshape((1,1))
    deltaw=np.concatenate((deltaw1,deltaw2,deltaw3),axis=0)#w gardient
    deltaw0=np.concatenate((deltaw01,deltaw02,deltaw03),axis=0)#bias gardient
    sigma1=pow(alpha*(sigma1**2)+(1-alpha)*(deltaw**2),0.5)
    sigma2=pow(alpha*(sigma2**2)+(1-alpha)*(deltaw0**2),0.5)
    w=w-(lr/sigma1)*deltaw
    w0=w0-(lr/sigma2)*deltaw0
#计算准确率
ztrain=np.concatenate((z1.reshape((-1,1)),z2.reshape((-1,1)),z3.reshape(-1,1)),axis=1)
ytrainpred=(ztrain.argmax(axis=1)).reshape((-1,1))
trainscore=sum(ytrainpred==ytrain)/xtrain.shape[0]
print('train score:',trainscore)
#计算测试集
seg1=np.exp(np.dot(xtest,w1.T)+w01)
seg2=np.exp(np.dot(xtest,w2.T)+w02)
seg3=np.exp(np.dot(xtest,w3.T)+w03)
total=seg1+seg2+seg3
zz1=seg1/total#class1
zz2=seg2/total#class2
zz3=seg3/total#class3
ztest=np.concatenate((zz1.reshape((-1,1)),zz2.reshape((-1,1)),zz3.reshape(-1,1)),axis=1)
ytestpred=(ztest.argmax(axis=1)).reshape((-1,1))
testscore=sum(ytest==ytestpred)/xtest.shape[0]
print('test score:',testscore)
