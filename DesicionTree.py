import numpy as np
import pandas as pd


row_data={'no surfacing':[1,1,1,0,0],
          'flippers':[1,1,0,1,1],
          'fish':['yes','yes','no','no','no']}
dataset=pd.DataFrame(row_data)

#ID3算法 ：决策分类树


#计算香农熵
def calEnt(dataset):
    n=dataset.shape[0]
    #标签的类别数
    class_num=dataset.iloc[:,-1].value_counts()
    p=class_num/n #每一类标签所占比
    ent=(-p*np.log2(p)).sum()

    return ent

#属性a有v个取值：a1,a2,……，av   Gain(D,a)=Ent(D)- \sum |Dv|/|D|  Ent(Dv)
def bestSplit(dataset):
    baseEnt=calEnt(dataset)#原始熵
    bestGain=0  #初始信息增益
    axis=-1  #初始最佳切分列
    for i in range(dataset.shape[1]-1):
        levels=dataset.iloc[:,i].value_counts().index  #提取出每一列的所有取值
        ents=0 #初始化子节点的信息熵
        for j in levels:
            childSet=dataset[dataset.iloc[:,i]==j]  #取出第i列为 j 的所有子数据集
            ent=calEnt(childSet)
            ents+=(childSet.shape[0]/dataset.shape[0])*ent #计算加权信息熵
        infoGain=baseEnt-ents  #当前列的信息增益
        if infoGain >bestGain:
            bestGain=infoGain
            axis=i
    return axis

#按照给定的列切分数据集
#dataset:原始数据集   axis:指定的索引列  value:指定的属性值
def mySplit(dataset,axis,value):
    col=dataset.columns[axis]  #找出最佳切分列的名字
    redataset=dataset.loc[dataset[col]==value,:].drop(col,axis=1) #按每一列取不同的值时，进行切分
    return redataset

#递归构建决策树
def createTree(dataset):
    fealist=list(dataset.columns) #数据集所有的列名
    classlist=dataset.iloc[:,-1].value_counts() #获取最后一列类标签
    #判读最多的标签数是否等于数据集行数，或者数据集是否只有一列
    #递归出口：现在要划分的数据集只有一列 (遍历完了所有特征列)  或  每个分支下所有的数据都属于同一类（这是叶子节点）
    if classlist[0]==dataset.shape[0] or dataset.shape[1]==1:
        return classlist.index[0]  #返回类标签
    axis=bestSplit(dataset) #确定最佳切分索引
    bestFeat=fealist[axis]  #索引对应的特征
    myTree={bestFeat:{}} #初始化字典  key 为特征  value 为 子树
    del fealist[axis]
    valueset=set(dataset.iloc[:,axis]) #最佳切分列所有属性值
    for value in valueset:  #每一个属性递归建立树
        myTree[bestFeat][value]=createTree(mySplit(dataset,axis,value))
    return myTree

#Tree=createTree(dataset)
#print(Tree)
#存储
#np.save('tree.npy',Tree)
#提取
#Tree=np.load('tree.npy').item()

#tree:已经构建好的决策树  labels：存储选择的最优特征标签  testVec:测试集数据
def classify(tree,labels,testVec):
    firstStr=next(iter(tree)) #决策树第一个节点
    second_dict=tree[firstStr] #下一个字典：子树
    featIndex=labels.index(firstStr) #第一个节点所在的列
    for key in second_dict.keys():
        if testVec[featIndex]==key:
            if type(second_dict[key])==dict:
                classLabel=classify(second_dict[key],labels,testVec)
            else:
                classLabel=second_dict[key]
    return classLabel

def acc_classify(train,test):
    tree=createTree(train)
    labels=list(train.columns)
    result=[]
    for i in range(test.shape[0]):
        testVec=test.iloc[i,:-1]
        classLabel=classify(tree,labels,testVec)
        result.append(classLabel)
    test['predict']=result #预测结果添加到最后一列
    acc=(test.iloc[:,-1]==test.iloc[:,-2]).mean()
    return acc,test


#CART算法  回归树
#寻找最佳切分列划分数据集；
# 二元切分法：若特征值大于给定值走左子树，否则右子树；节点不能再分则保存为叶节点
#叶节点为当前所有数据的均值
#构建模拟数据
x1=np.random.normal(loc=-10,scale=0.5,size=50)
x2=np.random.normal(loc=-10,scale=0.5,size=50)
xx=np.ones((100,)).reshape((-1,1))
y1=x1**2+2*x1
y2=x2**2+2*x2

x=np.concatenate([x1,x2],axis=0)
x=x.reshape((-1,1))
x=np.concatenate([xx,x],axis=1)
y=np.concatenate([y1,y2],axis=0).reshape((-1,1))

data=np.concatenate([x,y],axis=1)
data=pd.DataFrame(data)

#对每个特征值  1，将数据切分成两份（函数1） 2，计算切分误差（函数2）
# 若当前误差小于最小误差，则将当前切分为最佳切分,当前误差为最小误差
#返回最佳切分的特征和阈值

#切分数据函数
#data：原始数据    feature：待切分的特征索引  value：该特征的值
#返回   切分的两份数据集
def binSplitDataset(dataset,feature,value):
    data0=dataset.loc[dataset.iloc[:,feature]>value,:]
    data1=dataset.loc[dataset.iloc[:,feature]<=value,:]
    return data0,data1

#计算总方差
def errType(dataset):
    var=dataset.iloc[:,-1].var()*dataset.shape[0]
    return var

#生成叶节点
def leafType(dataset):
    leaf=dataset.iloc[:,-1].mean()
    return leaf

#最佳切分函数
#datset leafType errType  ops:用户定义的参数元祖:(允许的误差下降值，切分的最少样本数)
#返回：bestIndex:最佳切分列  bestValue:最佳切分值
def chooseBestSplit(dataset,leafType=leafType,errType=errType,ops=(1,4)):
    tols=ops[0]
    tolN=ops[1]
    #当前所有值相等，则退出
    if len(set(dataset.iloc[:,-1].values))==1:
        return None, leafType(dataset)
    m,n=dataset.shape
    #初始误差（未切分前）
    S=errType(dataset)
    bestS=np.inf
    bestIndex=0
    bestValue=0
    #遍历所有特征列
    for featIndex in range(n-1):
        colval=set(dataset.iloc[:,featIndex].values)
        #遍历该特征下的所有值
        for splitVal in colval:
            #根据当前值做数据切分
            mat0,mat1=binSplitDataset(dataset,featIndex,splitVal)
            #如果数据少于tolN,则退出
            if mat0.shape[0]<tolN or mat1.shape[0]<tolN:
                continue

            #切分后的误差
            newS=errType(mat0)+errType(mat1)
            #如果切分后的误差估计更小，则更新索引和特征值
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    #如果误差减少不大则退出
    if (S-bestS)<tols:
        return None,leafType(dataset)
    #根据最佳的切分特征和值切分数据集合
    mat0,mat1=binSplitDataset(dataset,bestIndex,bestValue)
    if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
        return None, leafType(dataset)
    return bestIndex,bestValue

#找到最佳切分特征
#若该节点不能再分，则保存为叶子节点
#执行二元切分  右子树  createTree   左子树  createTree

def createTree1(dataset,leafType=leafType,errType=errType,ops=(1,4)):
    #选择最佳切分特征和值
    col,val=chooseBestSplit(dataset,leafType,errType,ops)
    #如果没有特征，则返回特征值
    if col==None:
        return val
    #回归树
    reTree={}
    reTree['spInd']=col
    reTree['spVal']=val
    #分成左数据和右数据
    lset,rset=binSplitDataset(dataset,col,val)
    #创建左右子树
    reTree['left']=createTree1(lset,leafType,errType,ops)
    reTree['right'] = createTree1(rset, leafType, errType, ops)
    return reTree

tree=createTree1(data)
print(tree)





