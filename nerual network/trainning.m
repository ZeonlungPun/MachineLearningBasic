%%Radical Bias Function neural network 5
%trainning
function [c,w,sigma1,i,error]=trainning(x,y,iters,hidden_nums,tolerence_error,yita)
%yita:learning rate,i:max iters
[m,n]=size(x);
h=hidden_nums;
%initialize parameters
sigma1=unifrnd(0,1,h,1);
%c=unifrnd(-1,1,h,n);
c=[];%centers
for i=1:h
    for j=1:size(x,2)
        c1(1,j)=unifrnd(min(x(:,j)),max(x(:,j)),1,1);
    end
    c=[c1;c];
end
w=unifrnd(-10,10,h+1,1);
error=[];
for i=1:iters
%forward propagation
    hi_output=change(sigma1,x,c);%hidden layer output(m,h)
    yi_input=addintercept(hi_output);%output layer input(m,h+1)
    yi_output=yi_input*w;%output layer predict(m,1)
    error1=calMSE(yi_output,y);
    error=[error;error1];
    if error< tolerence_error
        break
    end
%backward propagation
    deltaw=yi_input'*(yi_output-y)./m;%(h+1,m)x(m,1)
    w=w-yita*deltaw;
    deltasigma=((hi_output.*l2norm(x,c))'*(yi_output-y)).*w(2:end,:)./(m*sigma1.^3);
%(h,m)x(m,1)
    sigma1=sigma1-yita*deltasigma;
    deltac1=w(2:end,:)./(sigma1.^2);%(h,1)
    deltac2=zeros(1,n);%(1,n)
    for j=1:m
        tt=yi_output-y;
        hh=unidrnd(h,1,1);
        deltac2=deltac2+tt(j)*(hi_output(j)*(x(j,:)-c(hh,:)));
    end
    deltac=deltac1*deltac2;%(h,1)x(1,n)
    c=c-yita*deltac./m;
end
return

