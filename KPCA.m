%KPCA with wine/breast cancer datasets
%data=xlsread('breast_cancer.xlsx','A2:AE512');
data=xlsread('wine.xlsx','A2:N179');
x=data(:,1:end-1);
x=zscore(x);%标准化
y=data(:,end);
[m,n]=size(x);%返回样本的样本数和特征数
x=x';%以列为样本，行为一个特征维度
k=[];%核函数化后的矩阵
norm=[];%储存范数
sigma=10;%超参数
for i=1:length(x)
    for j=1:length(x)
        norm2=(x(:,i)-x(:,j))'*(x(:,i)-x(:,j));
        norm(i,j)=norm2;
        k(i,j)=exp(-norm2./(2*sigma^2));
    end
end
ln=ones(m,1)*ones(m,1)';
k_c=k-k*ln-ln*k+ln*k*ln;%将k中心化
[alpha,lamda]=eig(k_c);%求解k_c的特征值和特征向量
lamda=diag(lamda);%提取出特征值
alpha=alpha./((lamda').^0.5);%特征向量单位化
%提取2、3个主成分
figure(1)
t1=lamda(1)*alpha(:,1);
t2=lamda(2)*alpha(:,2);
t3=lamda(3)*alpha(:,3);
gscatter(t1,t2,y)%降维后作二维图
xlabel('t1')
ylabel('t2')
figure(2)
index0=find(y==0);
index1=find(y==1);
index2=find(y==2);
t11=t1(index0);
t21=t2(index0);
t31=t3(index0);
scatter3(t11,t21,t31,'r')
hold on
t12=t1(index1);
t22=t2(index1);
t32=t3(index1);
scatter3(t12,t22,t32,'b')
hold on
t13=t1(index2);
t23=t2(index2);
t33=t3(index2);
scatter3(t13,t23,t33,'y')
xlabel('t1')
ylabel('t2')
zlabel('t3')
legend('0','1','2')