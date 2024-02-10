#手写随机森林
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
#随机模拟一些数据
x,y=make_classification(n_samples=1000,n_features=30,n_informative=2,
                        n_redundant=0,random_state=0,shuffle=False)
data=pd.DataFrame(x)
data['label']=y#拼接数据
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)#划分测试集、训练集
#开始手写
#设置一些分类器的参数值
n_estimators=100#基分类器数量
criterion='gini'#特征提取标准
random_state=420
max_depth=5
max_feature='auto'
bookstrap=False
#每一个基分类器都需要一个randomstate
np.random.seed(random_state)
#利用随机数得到100个randomstate
random_seed_list=(np.random.rand(n_estimators)*n_estimators*100).astype(int)
#bookstrap=True,则有放回的随机抽样，sample设置为replace=true;
# 否则所有基分类器都采用一个样本
data_list=[]
if bookstrap:
    for i in range(n_estimators):
        a=data.sample(data.shape[0],replace=True)
        data_list=data_list.append(a)
else:
    data_list=[data]*n_estimators

#开始构建随机森林
estimators=[DecisionTreeClassifier( criterion=criterion,
                                    max_depth=max_depth,
                                    random_state=i,
                                    max_features=max_feature,).fit(xtrain,ytrain)
for i in random_seed_list]#将构建好的基分类器都传入列表中，列表解析式
#预测数据
#所有基分类器都对数据做一下预测，并提取至列表中
predict_log_list1=np.array([i.predict_proba(xtrain) for i in estimators])#训练集
predict_log_list2=np.array([i.predict_proba(xtest) for i in estimators])#测试集
#求平均数
predict_proba1=predict_log_list1.mean(axis=0)
predict_proba2=predict_log_list2.mean(axis=0)
#将最大值所对应的类别提取出来，此即为预测值
predict1=predict_proba1.argmax(axis=1)#最大值索引
predict2=predict_proba2.argmax(axis=1)#最大值索引
#计算测试集准确率
accuracy1=sum(predict1==ytrain)/xtrain.shape[0]
accuracy2=sum(predict2==ytest)/xtest.shape[0]
print(accuracy1,accuracy2)#测试集准确略。预测集准确率



