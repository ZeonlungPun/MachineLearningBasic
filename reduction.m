%reduction methods 
%pca with SVD:Singular value decomposition
data=xlsread("iris.xlsx",'A2:E151');
x=data(:,1:end-1);
y=data(:,end);
xx=zscore(x);
[u,s,v]=svd(xx);
new=xx*v(:,1:2);
figure(1)
gscatter(new(:,1),new(:,2),y);

%NMS:Non-negative matrix factorization
xx1=(x-min(x))./(max(x)-min(x));
V=xx1';
k=2;
[n,m]=size(V);
h=rand(k,m);
w=rand(n,k);
epochs=2000;
for i=1:epochs
    h=h.*(w'*V)./(w'*w*h);
    w=w.*(V*h')./(w*h*h');
end
new1=h';
figure(2)
gscatter(new1(:,1),new1(:,2),y)

%MDS:Multi dimension scaling
gram=xx1*xx1';
h=repmat((diag(gram))',size(xx1,1),1);
dist=(h+h'-2*gram).^0.5;
D=dist.^2;
In=ones(m,m);
I=ones(m,1);
hh=In-I*I'./m;
B=-0.5*hh*D*hh;
[u1,s1,v1]=svd(xx1);
new2=xx1*v1(:,1:2);
figure(3)
gscatter(new2(:,1),new(:,2),y)



