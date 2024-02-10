clear
data=xlsread('iris.xlsx','A2:E151');
xraw=data(:,1:end-1);
x=(xraw-min(xraw))./(max(xraw)-min(xraw));%归一化
data1=[ones(length(data),1),x,data(:,end)+1];
%w1=unifrnd(-15,15,size(xraw,2)+1,3);%规定初始值
w1=ones(size(xraw,2)+1,3);
w=w1';
%w01=ones(size(xraw,2)+1,1);
%w02=ones(size(xraw,2)+1,1);
%w03=ones(size(xraw,2)+1,1);

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
%准备工作:算出初始值
%用矩阵乘法的形式计算，节约时间
w01=w(1,:)';%更新好的各类别权重
w02=w(2,:)';
w03=w(3,:)';
pt=exp(xtrain*w01)+exp(xtrain*w02)+exp(xtrain*w03);
p1=exp(xtrain*w01)./pt;
p2=exp(xtrain*w02)./pt;
p3=exp(xtrain*w03)./pt;
value=[p1,p2,p3];
[value1,index]=max(value');
per=index';%当前预测类别
y=zeros(length(xtrain),3);
for i=1:length(xtrain);
    label=per(i);
    y(i,label)=1;%初始输出类别矩阵
end
yita=0.01;%步长
num=100000;%迭代次数
stopvalue=0.00001;%停止迭代条件
ww=[];


for j=1:num
    %小批量梯度下降
    sample=[xtrain,value,y];
    rr=randperm(length(sample));
    sample_extract=sample(rr(1:round(0.25*length(sample))),:);%只提取0.25的训练集做梯度下降
    xx=sample_extract(:,1:end-6);
    yy=sample_extract(:,end-2:end);
    vv=sample_extract(:,end-5:end-3);
    grad=-((yy-vv)'*xx)./length(data);%梯度迭代
    w=w+yita*grad;%权重迭代
    w01=w(1,:)';%更新好的各类别权重
    w02=w(2,:)';
    w03=w(3,:)';
    ww=[w01';ww];
    %while w00*w00'<stopvalue%达到条件则迭代停止
     %   break
    %end
    %新的各类别概率
    pt=exp(xtrain*w01)+exp(xtrain*w02+exp(xtrain*w03));
    p1=exp(xtrain*w01)./pt; 
    p2=exp(xtrain*w02)./pt;
    p3=exp(xtrain*w03)./pt;
    value=[p1,p2,p3];
    [value1,index]=max(value');
    per=index';%当前预测类别
    y=zeros(length(xtrain),3);
    for i=1:length(xtrain);
        label=per(i);
        y(i,label)=1;%输出类别矩阵
    end
end
trainpre=per;
trainscore=sum(trainpre==train(:,end))./length(xtrain)%训练集分数

%测试集结果
pt=exp(xtest*w01)+exp(xtest*w02+exp(xtest*w03));
p1=exp(xtest*w01)./pt;
p2=exp(xtest*w02)./pt;
p3=exp(xtest*w03)./pt;
value=[p1,p2,p3];%预测为3个类别的概率（列数表示类别）
[value1,index]=max(value');
per=index';%当前预测类别
testpre=per;
testscore=sum(testpre==test(:,end))./length(xtest)%测试集分数
plot(1:num,ww)