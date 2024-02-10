data=xlsread('wine.xlsx','B2:O179');
xraw=data(:,1:end-1);
x=(xraw-min(xraw))./(max(xraw)-min(xraw));%归一化
data1=[ones(length(data),1),x,data(:,end)+1];
w01=unifrnd(-1,1,size(xraw,2)+1,1);%规定初始值
w02=unifrnd(-1,1,size(xraw,2)+1,1);%规定初始值
w03=unifrnd(-1,1,size(xraw,2)+1,1);%规定初始值
%w01=ones(size(xraw,2)+1,1);
%w02=ones(size(xraw,2)+1,1);
%w03=ones(size(xraw,2)+1,1);
w=[w01';w02';w03'];%初始权重矩阵
numtestratio=0.25;%测试集占比
r=randperm(length(data));%根据样本数量生成随机数索引
num_test=round(length(data)*numtestratio);
test=data1(r(1:num_test),:);%测试集数据
r(1:num_test)=[];%用过的索引清空
train=data1(r,:);%训练集
xtrain=train(:,1:end-1);
ytrain=train(:,end);
xtest=test(:,1:end-1);
ytest=test(:,end);
%准备工作

per=[];
value=[];
for i =1:length(xtrain)%算出初始值
    pt=exp(-xtrain(i,:)*w01)+exp(-xtrain(i,:)*w02+exp(-xtrain(i,:)*w03));
    p1=exp(-xtrain(i,:)*w01)./pt;
    p2=exp(-xtrain(i,:)*w02)./pt;
    p3=exp(-xtrain(i,:)*w03)./pt;
    val=[p1,p2,p3];
    [value1,index]=max(val');
    per=[per;index];%当前预测类别
    value=[value;val];%预测为3个类别的概率（列数表示类别）
end
y=zeros(length(xtrain),3);
for i=1:length(xtrain);
    label=per(i);
    y(i,label)=1;%初始输出类别矩阵
end

yita=0.01;%步长
num=50000;%迭代次数
for j=1:num
    grad=-((y-value)'*xtrain)./length(data);%梯度迭代
    w=w+yita*grad;%权重迭代
    w01=w(1,:)';%更新好的各类别权重
    w02=w(2,:)';
    w03=w(3,:)';
    value=[];
    per=[];
    for i=1:length(xtrain);%循环计算每一个样本属于各类别的概率
        pt=exp(-xtrain(i,:)*w01)+exp(-xtrain(i,:)*w02+exp(-xtrain(i,:)*w03));
        p1=exp(-xtrain(i,:)*w01)./pt;
        p2=exp(-xtrain(i,:)*w02)./pt;
        p3=exp(-xtrain(i,:)*w03)./pt;
        val=[p1,p2,p3];%每一个样本属于各类别的概率
        [value1,index]=max(val');
        per=[per;index];%当前预测类别
        value=[value;val];%预测为3个类别的概率（列数表示类别）
    end
    y=zeros(length(xtrain),3);
    for i=1:length(xtrain);
        label=per(i);
        y(i,label)=1;%输出类别矩阵
    end
end
trainpre=per;
trainscore=sum(trainpre==train(:,end))./length(xtrain)
value=[];
per=[]
 for i =1:length(xtest)%测试集结果
    pt=exp(-xtest(i,:)*w01)+exp(-xtest(i,:)*w02+exp(-xtest(i,:)*w03));
    p1=exp(-xtest(i,:)*w01)./pt;
    p2=exp(-xtest(i,:)*w02)./pt;
    p3=exp(-xtest(i,:)*w03)./pt;
    val=[p1,p2,p3];
    [value1,index]=max(val');
    per=[per;index];%当前预测类别
    value=[value;val];%预测为3个类别的概率（列数表示类别）
 end
testpre=per;
testscore=sum(testpre==test(:,end))./length(xtest)

