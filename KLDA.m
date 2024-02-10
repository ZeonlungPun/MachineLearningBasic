%kernel LDA for 2 classes
%data=xlsread('dbscandata.xlsx','B2:D201');
data=xlsread('circles_datasets1.xlsx','B2:D1001');
x=data(:,1:end-1);
y=data(:,end);
index1=find(y==0);
index2=find(y==1);
n1=size(index1,1);
n2=size(index2,1);
x1=x(index1,:);
x2=x(index2,:);
sigma=0.4;%超參數
m1=[];
for i=1:size(x,1)
    for j=1:size(x1,1)
        norm2=(x(i,:)-x1(j,:))*(x(i,:)-x1(j,:))';
        m1(i,j)=exp(-norm2./(2*sigma^2));
    end
end
m1=sum(m1,2)./n1;
m2=[];
for i=1:size(x,1)
    for j=1:size(x2,1)
        norm2=(x(i,:)-x2(j,:))*(x(i,:)-x2(j,:))';
        m2(i,j)=exp(-norm2./(2*sigma^2));
    end
end
m2=sum(m2,2)./n2;
m=(m2-m1)*(m2-m1)';
k1=m1.*n1;
k2=m2.*n2;
n=k1*k1'-(k1*k1')./n1+k2*k2'-(k2*k2')./n2;
n=n+0.01*eye(size(n,1));
%m=m+0.01*eye(size(m,1));
[d,lamda]=eig(inv(n)*m);
w1=d(:,1);
w2=d(:,2);
z1=w1.*m1;
z2=w2.*m2;
gscatter(z1,z2,y)
