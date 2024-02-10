%back propagation neural network
%main code
data=xlsread('boston.xlsx','B2:O507');
%data1=[data11',data12'];
%data=xlsread('dd.xlsx','A2:F123');
%data1=zscore(data);
data1=(data-min(data))./(max(data)-min(data));
ratio=0.3;%test ratio
num=round(length(data1)*ratio);%testing set number
rr=randperm(length(data1));
test=data1(1:num,:);%testing set
train=data1(num+1:end,:);%trainning set
x1=train(:,1:end-1);
y1=train(:,end);
x2=test(:,1:end-1);
y2=test(:,end);
[m,n]=size(x1);
hiddenneurons1=15;%hidden layer1 nums
hiddenneurons2=15;%hidden layer2 nums
%initialize weigth
w_xinput=initw(n,hiddenneurons1);%input layer to hidden layer1
w_hidden1=initw(hiddenneurons1,hiddenneurons2);%hidden layer1 to hidden layer2
w_hidden2=initw(hiddenneurons2,1);%hidden layer2 to output layer
r=0.05;%learnning rate
iters=15000;%max iterations
tolerance=10^-4;
%trainning set trainning
[w_hidden1,w_hidden2,w_xinput,errorlist]=bptrainning(x1,y1,r,iters,tolerance,w_xinput,w_hidden1,w_hidden2);
%trainning set predicting
y1predict=bppredicting(x1,w_xinput,w_hidden1,w_hidden2);
mse1=calMSE(y1,y1predict)
%testing set predicting
y2predict=bppredicting(x2,w_xinput,w_hidden1,w_hidden2);
mse2=calMSE(y2,y2predict)
%result visualize
figure(1)
plot(1:length(errorlist),errorlist')
axis([-1000,10000,min(errorlist),max(errorlist)])
title('The trainning set error(mse) changes with the number of iterations')
xlabel('the number of iterations')
ylabel('MSE')
figure(2)
plot(1:length(train),[y1';y1predict'])
legend('actual value','predicted value')
title('back propagation neural network;trainning set performance ')
xlabel('sample number')
ylabel('value')
figure(3)
plot(1:length(test),[y2';y2predict'])
legend('actual value','predicted value')
title('back propagation neural network;testing set performance ')
xlabel('sample number')
ylabel('value')