%Canonical Correlation Analysis
data=xlsread('alldata.xlsx','A2:P8372');
x=data(:,1:9);
y=data(:,10:end);
x=zscore(x);
y=zscore(y);
p=size(x,2);
q=size(y,2);
num=size(x,1);%样本数
r=min(p,q);
sigmaxx=corrcoef(x);
sigmayy=corrcoef(y);
sigmaxy=[];
sigmayx=[];
for i=1:size(x,2)
    for j=1:size(y,2)
        co=corrcoef(x(:,i),y(:,j));
        sigmaxy(i,j)=co(1,2);
    end
end
for i=1:size(y,2)
    for j=1:size(x,2)
        co=corrcoef(x(:,j),y(:,i));
        sigmayx(i,j)=co(1,2);
    end
end
m1=inv(sigmaxx)*sigmaxy*inv(sigmayy)*sigmayx;
m2=inv(sigmayy)*sigmayx*inv(sigmaxx)*sigmaxy;
[p1,lamda1sq]=eig(m1);
[p2,lamda2sq]=eig(m2);
%典型相关系数阵
a=p1(:,1:r);
b=p2(:,1:r);
lmada1=(diag(lamda1sq)).^0.5;
lmada2=(diag(lamda2sq)).^0.5;
%计算u\v得分（各样本典型相关得分）
u=[];
v=[];
for i=1:r
    u1=sum(a(:,i)'.*x,2);
    v1=sum(b(:,i)'.*y,2);
    u=[u,u1];
    v=[v,v1];
end
%整体检验统计量
alpha=0.05;%显著性水平
gama1=prod(1-lmada1);
q1=-(num-0.5*(p+q+3))*log(gama1);
df1=p*q;
if q1>= chi2cdf(alpha,df1)%根据置信水平alpha反查临界值
    disp('至少一对典型变量之间相关性显著')
else
    disp('整体检验不通过')
end
for k=2:r
    gama1=prod(1-lmada1(k:end));
    q1=-(num-k-0.5*(p+q+3)+sum((lmada1(:,k:end)).^0.5))*log(gama1);
    df1=(p-k)*(q-k);
    if q1>= chi2cdf(alpha,df1)%根据置信水平alpha反查临界值
        fprintf('第%d对典型变量之间相关性显著',k,'\n')
    else
        fprintf('第%d对典型变量检验不通过',k,'\n')
    end
end


roxu=[];
roxv=[];
royu=[];
royv=[];
%典型结构分析
for i=1:size(x,2)
    for j=1:size(u,2)
        co1=corrcoef(x(:,i),u(:,j));
        co2=corrcoef(x(:,i),v(:,j));
        roxu(i,j)=co1(1,2);
        roxv(i,j)=co2(1,2);
    end
end
for i=1:size(y,2)
    for j=1:size(u,2)
        co1=corrcoef(y(:,i),u(:,j));
        co2=corrcoef(y(:,i),v(:,j));
        royu(i,j)=co1(1,2);
        royv(i,j)=co2(1,2);
    end
end
%x/y被u/v解释的方差比例
mu=sum(roxu.^2)/p;
mv=sum(roxv.^2)/p;
nu=sum(royu.^2)/q;
nv=sum(royv.^2)/q;
