%随机模拟一些点作分类用途
random=unifrnd(-1,1,15,2);
group1=ones(15,2)+random;
group2=2.65*ones(15,2)+random;
x=[group1;group2]
data=[group1,-1*ones(15,1);group2,1*ones(15,1)];%数据与对应标签
y=data(:,end)%真实类别
%感知机,单样本梯度下降，二维
count=0%迭代次数统计
lam=0.1%步长
w=ones(size(x,2),1);%初始值
b=1;
while 1
    count=count+1;
    missing_index=-1;%初始化误分类点的索引
    for i=1:length(data)
        checking=y(i).*(x(i,:)*w+b);
        if checking <=0%%找到误分类点
            missing_index=i;
            break%每次循环只找一个误分类点,找到后for循环不再执行
        end
    end
    if missing_index==-1%若没有误分类点，表示感知机已构建完成,跳出while循环
            break
    end
    w=w+lam*y(missing_index).*x(missing_index,:)';%每次只用一个误分类点进行迭代
    b=b+lam*y(missing_index);
    
end
%画出点以及对应的超平面
scatter(group1(:,1),group1(:,2),'red');
hold on
scatter(group2(:,1),group2(:,2),'blue');
hold on

k=-w(1)./w(2);%将直线改写成斜截式便于作图
bb=-b./w(2);
xx=0:4;
yy=k.*xx+bb;
plot(xx,yy,'--')
title('perceptron')
xlabel('dimension1')
ylabel('dimension2')
legend('group1','group2','separating hyperplane')

    

