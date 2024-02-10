%linear discriminant analysis for 2 classification
data=xlsread('breast_cancer.xlsx','A2:AE570');
x=data(:,1:end-1);
x=(x-min(x))./(max(x)-min(x));
y=data(:,end);
index1=find(y==0);
index2=find(y==1);
n1=size(index1,1);
n2=size(index2,1);
mu1=sum(x(index1,:))./n1;
mu1=mu1';
mu2=sum(x(index2,:))./n2;
mu2=mu2';
%算协方差
cov1=0;
cov2=0;
for i=1:n1
    cc=(x(i,:)'-mu1)* (x(i,:)'-mu1)';
    cov1=cov1+cc;
end
for i=1:n2
    cc=(x(i,:)'-mu2)* (x(i,:)'-mu2)';
    cov2=cov2+cc;
end
sw=cov1+cov2;
sb=(mu1-mu2)*(mu1-mu2)';
[d,lamda]=eig(inv(sw)*sb);
w1=d(:,1);
w2=d(:,4);
z1=x*w1;
z2=x*w2;
gscatter(z1,z2,y)
%利用降维后的数据作分类判别(只用一维)
z11=mean(x(index1,:))*w1;
z12=mean(x(index2,:))*w1;
zbar=(z11+z12)./2;
judge=(z1>zbar);
acc=sum(judge==y)./length(y)