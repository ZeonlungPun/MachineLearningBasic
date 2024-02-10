#logistic regression case
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
# data=pd.read_excel('breast_cancer.xlsx',engine='openpyxl')
# x=data.iloc[:,0:30]
data=pd.read_excel('processed.xlsx',engine='openpyxl')
c1=data.loc[data['y']==1]
c2=data.loc[data['y']==0]

c21=c2.iloc[10000::,:]

c2=c2.iloc[0:10000,:]

data=pd.concat([c1,c2],axis=0)
x=data.iloc[:,0:-1]
print(x)

(m,n)=x.shape
y=data.iloc[:,-1]

#将DataFrame转化为numpy数组，方便作矩阵运算
x=np.array(x)
one=np.ones((x.shape[0],1))
x=x.reshape(m,n)

#ptest=np.squeeze(ptest,axis=1)

#x=np.concatenate((one,x),axis=1)

x=x.reshape(m,n)
y=np.array(y)
y=y.reshape(m,1)
print(y)
random_number=np.random.permutation(m)#产生随机数序列

#random_number=random_number[:,np.newaxis]#升维操作，以使其和x，y维度对应
random_number.reshape(m,1)

random_num=round(m*0.2)#test set number
z11=random_number[1:random_num]#测试集所对应的样本序号
z12=random_number[(random_num+1)::]#训练集所对应的样本序号
xtest=x[z11,:]
ytest=y[z11,:]
xtrain,ytrain=x[z12,:],y[z12,:]
#rus = RandomUnderSampler(random_state=0)
#xtrain,ytrain = rus.fit_resample(xtrain, ytrain)

xtrain=(xtrain-np.min(xtrain,axis=0))/(np.max(xtrain,axis=0)-np.min(xtrain,axis=0)+0.01)
xtrain=np.concatenate([np.ones((xtrain.shape[0],1)),xtrain],axis=1)


epochs=800#iteration
w=np.ones((n+1,1))#设定初始值
#gardient descend
for i in range(epochs+1):
    gradient=np.zeros((n+1,1))
    for j in range(xtrain.shape[0]):
        p=1/(1+np.exp(np.dot(-xtrain[j].reshape(1,n+1),w)))
        grad=(ytrain[j]-p)*xtrain[j]
        grad=grad.T
        gradient=gradient+grad
    w=w+0.02*gradient#updata the gradient

ptrain=1/(1+np.exp(-np.dot(xtrain,w)))#train probability

xtest=(xtest-np.min(xtest,axis=0))/(np.max(xtest,axis=0)-np.min(xtest,axis=0)+0.01)
xtest=np.concatenate([np.ones((xtest.shape[0],1)),xtest],axis=1)


ptest=1/(1+np.exp(-np.dot(xtest,w)))#test probability

pred1=np.array(ptrain>0.5).reshape(-1,1)#布尔运算，大于阈值0.6判别为（True），等同于1类，否则为false（0类）
pred2=np.array(ptest>0.5).reshape(-1,1)
#预测值和真实值作布尔运算
trainscore=np.sum(pred1==ytrain.reshape(-1,1))/xtrain.shape[0]
testscore=np.sum(pred2==ytest.reshape(-1,1))/xtest.shape[0]
print('train score:',trainscore)
print('test score:',testscore)
f11 = f1_score(ytrain, pred1)
f12 = f1_score(ytest, pred2)
pc1 = precision_score(ytrain, pred1)
pc2 = precision_score(ytest, pred2)
rc1 = recall_score(ytrain, pred1)
rc2 = recall_score(ytest, pred2)
print('f1:',f11,f12)
print('precesion:',pc1,pc2)
print('recall:',rc1,rc2)


xx=c21.iloc[:,0:-1]
yy=c21.iloc[:,-1]
xx=np.array(xx)
yy=np.array(yy)
xx=np.concatenate([np.ones((xx.shape[0],1)),xx],axis=1)
xx=(xx-np.min(xx,axis=0))/(np.max(xx,axis=0)-np.min(xx,axis=0)+0.01)



pp=1/(1+np.exp(-np.dot(xx,w)))
pred=np.array(pp<0.5).reshape(-1,1)
print(pp)
print(yy)
score=np.sum(pred==yy.reshape(-1,1))/xx.shape[0]
print('score:',score)

#ptrain=np.squeeze(ptrain,axis=1)#降维，以匹配尺寸
#ptest=np.squeeze(ptest,axis=1)
"""
df1=pd.DataFrame(ptrain)
df2=pd.DataFrame(ptest)
predict1=pd.DataFrame()
predict2=pd.DataFrame()

for i in range(ptrain.shape[0]):
    datalist=0#引入临时存储空间
    if df1.iloc[i,0]>0.6:
        datalist=1
    else:
        datalist=0
    dd=pd.DataFrame([datalist])
    predict1=pd.concat([predict1,dd])
print(predict1)

for i in range(ptest.shape[0]):
    datalist=[]#引入临时存储空间
    if df2.iloc[i,0]>0.6:
        datalist.append(1)
    else:
        datalist.append(0)
    dd=pd.DataFrame(datalist)
    predict2=pd.concat([predict2,dd])
    
#ytrain=np.squeeze(ytrain,axis=1)
print('ytest:',ytest)
print(predict1)
ytest=np.squeeze(ytest,axis=1)
print(ytest)
trainscore=((predict1==ytrain).value_counts[True])/predict1.shape[0]
testscore=((predict2==ytest).value_counts()[True])/predict2.shape[0]
"""






