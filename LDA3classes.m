%linear discriminant analysis for 3 classes
data=xlsread('wine.xlsx','A2:N179');
x=data(:,1:end-1);
x=(x-min(x))./(max(x)-min(x));
y=data(:,end);
k=length(unique(y));%类数
index1=find(y==0);
index2=find(y==1);
index3=find(y==2);
n1=size(index1,1);
n2=size(index2,1);
n3=size(index3,1);
mu1=sum(x(index1,:))./n1;
mu1=mu1';
mu2=sum(x(index2,:))./n2;
mu2=mu2';
mu3=sum(x(index3,:))./n3;
mu3=mu3';
mu=(mean(x))';
sb=n1.*((mu1-mu)*(mu1-mu)'+(mu2-mu)*(mu2-mu)'+(mu3-mu)*(mu3-mu)');
%算协方差
cov1=0;
cov2=0;
cov3=0;
for i=1:n1
    cc=(x(i,:)'-mu1)* (x(i,:)'-mu1)';
    cov1=cov1+cc;
end
for i=1:n2
    cc=(x(i,:)'-mu2)* (x(i,:)'-mu2)';
    cov2=cov2+cc;
end
for i=1:n3
    cc=(x(i,:)'-mu3)* (x(i,:)'-mu3)';
    cov3=cov3+cc;
end
sw=cov1+cov2+cov3;
[d,lamda]=eig(inv(sw)*sb);
w1=d(:,1);
w2=d(:,2);
z1=x*w1;
z2=x*w2;
gscatter(z1,z2,y)
