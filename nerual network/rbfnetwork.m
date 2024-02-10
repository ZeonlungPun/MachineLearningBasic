%%Radical Bias Function neural network 6
%main code
data=xlsread('boston.xlsx','B2:O507');
%data1=data(:,1:end-1);
%data2=(data1-min(data1))./(max(data1)-min(data1));
%data=[data2,data(:,end)];
data=(data-min(data))./(max(data)-min(data));
ratio=0.3;
rr=randperm(size(data,1));
num=round(ratio*length(data));
test=data(1:num,:);
train=data(num+1:end,:);
xtrain=train(:,1:end-1);
ytrain=train(:,end);
xtest=test(:,1:end-1);
ytest=test(:,end);
iters=25000;
tolerence_error=0.01;
yita=0.025;
hidden_nums=20;
[c,w,sigma1,i,error]=trainning(xtrain,ytrain,iters,hidden_nums,tolerence_error,yita)
%trainning set prediction
h_output1=change(sigma1,xtrain,c);%hidden layer output(m,h)
yi_input1=addintercept(h_output1);%output layer input(m,h+1)
predicting1=yi_input1*w;%output layer predict(m,1)
mse1=calMSE(predicting1,ytrain)
%testing set prediction
h_output2=change(sigma1,xtest,c);%hidden layer output(m,h)
yi_input2=addintercept(h_output2);%output layer input(m,h+1)
predicting2=yi_input2*w;%output layer predict(m,1)
mse2=calMSE(predicting2,ytest)
figure(1)
plot(1:length(test),[ytest';predicting2'])
legend('actual value','predicted value')
title('Radical Bias Function neural network;testing set performance ')
xlabel('sample number')
ylabel('value')
figure(2)
plot(1:length(error),error)
axis([0,2500,-0.05,0.5])
xlabel('iteration number')
ylabel('error')
title('Radical Bias Function neural network;changing error ')