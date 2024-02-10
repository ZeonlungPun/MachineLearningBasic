%LASSO regression 
%generating simulation data
mu1=[150 150 150];
sigma1=10*ones(3,3);
%real variables
x=mvnrnd(mu1,sigma1,150);
mu2=zeros(1,6);
sigma2=ones(6,6);
%fake variables
x_fake=mvnrnd(mu2,sigma2,150);
xx=[x,x_fake];
noise=mvnrnd(0,1,150);
y=10*x(:,1)+35*x(:,2)-23*x(:,3)+noise;
%standardlize
xx=zscore(xx);
y=zscore(y);
L=1.75;%Lipschitz coeffient
lamda=1;% regularization coeffient
m=size(xx,2);% dimension of data
n=size(xx,1);%sample number of data
w=10*rand(m,1);%innitialize coeffient
z=[];
epochs=100;
times=0;
while times<=epochs
    %one iteraition
    for j=1:m%loop the dimension
        partial_gw=0;
        for i=1:n% calculate the gradient using all samples
            partial_gw=partial_gw+(y(i,:)-xx(i,:)*w).*xx(i,j);
        end
           partial_gw=-2*partial_gw./n;
           %undate parameters:w
           z(j)=w(j)-partial_gw./L;
           w(j)=sign(z(j)).*max(abs(z(j))-lamda/L,0);  
    end
    times=times+1;
end
     
    
    





