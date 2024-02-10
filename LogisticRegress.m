%logistics regresion algorithm
data1=xlsread('breast_cancer','A2:AF570');%读取全部数据
x=data1(:,1:end-1)%提取自变量
%xx=(x-min(x))./(max(x)-min(x))%归一化[0,1]
xx=x;
data=[ones(length(data1),1),xx,data1(:,end)];
numtestratio=0.3;%测试集占比
r=randperm(length(data));%根据样本数量生成随机数索引
num_test=round(length(data)*numtestratio);
test=data(r(1:num_test),:);%测试集数据
r(1:num_test)=[];%用过的索引清空
train=data(r,:);%训练集
xtrain=train(:,1:end-1)
ytrain=train(:,end)
xtest=test(:,1:end-1)
ytest=test(:,end);
yita=0.002%步长
stop_value=0.001%迭代停止值
num=8000%迭代次数
w=[];%记录权重变化
w0=ones(size(data(:,1:end-1),2),1)%规定初始值
%batch gardient descent
for i=1:num
    gradient=zeros(size(data1,2),1);%临时变量，每一次都要重新计算
    for j=1:length(xtrain);
        p=1./(1+exp(-xtrain(j,:)*w0));
        grad=(ytrain(j,1)-p)*xtrain(j,:)';
        gradient=gradient+grad;%计算每一次的梯度，批量梯度下降法
    end
        w0=w0+yita*gradient;%迭代梯度
        w=[w,w0];%为了可视化每一次W的迭代结果
     if (gradient'*gradient).^0.5<stop_value
         break
     end
end
p1=1./(1+exp(-xtrain*w0));
p2=1./(1+exp(-xtest*w0));
ratio=tabulate(data1(:,end));
p0=ratio(end,3)./100%阈值的设定
ytrainperdict=( p1>p0);
ytestperdict=(p2>p0)
yy1=(ytrain==ytrainperdict);
yy2=(ytest==ytestperdict);
trainscore=sum(yy1)./length(ytrainperdict)%训练集分数
testscore=sum(yy2)./length(ytestperdict)%测试集分数
plot(1:size(w,2),w)%学习曲线1：各权重随着迭代次数的增加的变化情况
title('学习曲线1：各权重随着迭代次数的增加的变化情况')

