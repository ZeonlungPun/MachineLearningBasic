%矩阵分解，推荐系统
%0表示数据缺失，需要预测
%mxn => mxk kxn m:user num n:items num k :latent vector dim
clear;
data=[1.4,0,1.1,0.7,0;0,0.3,0,0.7,0.5;0.4,0.3,0,0,0.3;1.4,0,1.2,0,0.8];
[m,n]=size(data);
%funk-svd
%初始化pi qi
k=50;
p=rand(m,k);
q=rand(n,k);
epochs=1000;%迭代次数
yita=0.01;%学习率
lamda=0.01;%正则化系数
[x0,y0]=find(data==0);%未知评分的坐标
for k=1:epochs
    %更新每一个pi和qi
    for i=1:m
        for j=1:n
            if i~=x0(i) && j~=y0(j)
                e=data(i,j)-p(i,:)*q(j,:)';
                q(j,:)=q(j,:)+2*yita*(e*p(i,:)-lamda*q(j,:));
                p(i,:)=p(i,:)+2*yita*(e*q(j,:)-lamda*p(i,:));
            end
        end
    end
end
pred=p*q';
num=find(data~=0);
mse=mean((data(num)-pred(num)).^2);
%svd
[u,s,v]=svd(data);
%截断奇异值的个数
t=2;
u1=u(:,1:t);
s1=s(1:t,1:t);
v1=v(:,1:t);
data1=u1*s1*v1';
mse1=mean((data(num)-data1(num)).^2);
%baline estimate
k=25;
p=rand(m,k);
q=rand(n,k);
bm=rand(m,1);
bn=rand(n,1);
epochs=1000;%迭代次数
yita=0.001;%学习率
lamda=0;%正则化系数

mu=sum(data)./sum(data>0);%每件物品已有评分的均值
for k=1:epochs
    %更新每一个pi和qi
    for i=1:m
        for j=1:n
            if i~=x0(i) && j~=y0(j)
                e=data(i,j)-p(i,:)*q(j,:)'-bm(i)-bn(j)-mu(j);
                q(j,:)=q(j,:)+2*yita*(e*p(i,:)-lamda*q(j,:));
                p(i,:)=p(i,:)+2*yita*(e*q(j,:)-lamda*p(i,:));
                bm(i)=bm(i)+2*yita*e-2*lamda*bn(j);
                bn(j)=bn(j)+2*yita*e-2*lamda*bm(i);
            end
        end
    end
end
pred1=p*q'+bm+bn';
num=find(data~=0);
mse2=mean((data(num)-pred1(num)).^2);

