%支持向量机support vector machine%软间隔soft margin
%二次规划问题
%随机模拟一些点作分类用途
random=unifrnd(-1,1,50,2);
group1=ones(50,2)+random;
group2=2.85*ones(50,2)+random;
x=[group1;group2];
data=[group1,-1*ones(50,1);group2,1*ones(50,1)];%数据与对应标签
y=data(:,end);%真实类别
h=[]%目标函数的H
for i=1:length(x)%对于所有样本都要遍历
    for j=1:length(x)
        h(i,j)=x(i,:)*(x(j,:))'*y(i)*y(j);
    end
end
f=-1*ones(length(x),1);%目标函数的f
%灯饰约束
aeq=y';
beq=zeros(1,1);
%不等式约束
ub=[];
ib=[];
%自变量约束
lb=zeros(length(x),1);
ub=[];
[a,fval]=quadprog(h,f,ib,ub,aeq,beq,lb,ub);%二次规划问题
w=0;%系数矩阵
u=0
for i=1:length(a)
    if a(i)<1e-05
        a(i)=0;
    end
end
ff=find(a~=0)
j=ff(1)%寻找a系数不等于的下标j
for i=1:length(x)%关键系数数求解
    w=w+a(i)*y(i)*x(i,:)';
    u=u+a(i)*y(i)*(x(i,:)*x(j,:)');
end
b=y(j)-u;
%画出点以及对应的超平面
scatter(group1(:,1),group1(:,2),'red');
hold on
scatter(group2(:,1),group2(:,2),'blue');
hold on
k=-w(1)./w(2);%将直线改写成斜截式便于作图
bb=-b./w(2);
xx=0:4;
yy=k.*xx+bb;
plot(xx,yy,'-')
hold on
yy=k.*xx+bb+1./w(2)
plot(xx,yy,'--')
hold on
yy=k.*xx+bb-1./w(2)
plot(xx,yy,'--')
title('support vector machine')
xlabel('dimension1')
ylabel('dimension2')
legend('group1','group2','separating hyperplane')
%分类决策函数作预测
predict=[];
for i=1:length(x)%预测第i个样本
    uu=0;%过度变量
    for j=1:length(x)%利用训练集的所有样本构建预测函数
        uu=uu+a(j)*y(j)*(x(j,:)*x(i,:)');
    end
    result=sign(uu+b);
    predict(i,1)=result;
end
judge=(predict==y);
score=sum(judge)./length(data)