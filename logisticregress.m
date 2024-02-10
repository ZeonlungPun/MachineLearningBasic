% Logistic regression with gradient descent
a1=a(:,1:5)%自变量
a1=zscore(a1)
y=a(:,6)%真实分类
x=[ones(40,1),a1]%数据矩阵,注意利用1矩阵要消除偏置
w=ones(6,1)%初始值
alpha=0.05%学习速率
grad=0
while abs(grad)>0.000001
    t=x*w
    yh=1./(1+exp(-t))
    error=y-yh
    grad=(x'*error)./40%梯度表达式
    w=w+alpha.*grad
end
for i=1:40
    if (1./(1+exp(-x(i,:)*w)))>0.5
        y1(i,1)=1
    else
        y1(i,1)=0
    end
end
b=(y1==y)
b1=sum(b==1)
b2=b1./40%预测的正确率