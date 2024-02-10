%back propagation neural network
data=xlsread('boston.xlsx','B2:O507');
[data11,ps1]=mapminmax(data(:,1:end-1)',-1,1);%normalise treatment
[data12,ps2]=mapminmax(data(:,end)',-1,1);
data1=[data11',data12'];

ratio=0.3;%test ratio
num=round(length(data)*ratio);%testing set number
rr=randperm(length(data));
test=data1(1:num,:);%testing set
train=data1(num+1:end,:);%trainning set
x1=train(:,1:end-1);
y1=train(:,end);
x2=test(:,1:end-1);
y2=test(:,end);

%initialise parameters
hidden_layer_neural_num=12;%hidden layer
output_layer_neural_num=1;
w1=0.5*rand(hidden_layer_neural_num,size(x1,2))-0.1;%first layer weight
w2=0.5*rand(output_layer_neural_num,hidden_layer_neural_num)-0.1;%output layer weight
b1=0.5*rand(hidden_layer_neural_num,1)-0.1;%bias1
b2=0.5*rand(output_layer_neural_num,1)-0.1;
%trainning start
iters=50000;%max trainning numbers
e0=6.5*10^(-3);%target error
for i=1:200
    %forward propagation
    a1=w1*x1'+repmat(b1,1,length(x1));%first layer to hidden layer
    o1=a1;%sigmoid activation function
    a2=w2*o1+repmat(b2,1,length(x1));%hidden layer to output layer
    o2=logsig(a2);
    o2=o2';
    error=o2-y1;%Error between predicted and measured value
    sse=sumsqr(error);% Error sum of squares
    if sse<e0
        break
    end
    %backward propagation with gradien descend
    %l2=(o2-y1).*(exp(-a2')./((1+exp(-a2')).^2));
    l2=(o2-y1).*logsig(a2').*(1-logsig(a2'));
    deltaw2=l2'*o1';
    deltab2=sum(error);
    w2=w2+0.35*deltaw2;%gradien descend
    b2=b2+0.35*deltab2;
    %l1=(l2*w2).*(exp(-a1')./((1+exp(-a1')).^2));
    l1=l2*w2.*logsig(a1').*(1-logsig(a1'));
    deltaw1=l1'*x1;
    deltab1=sum(l1)';
    b1=b1+0.35*deltab1;
    w1=w1+0.35*deltaw1;
end
figure(1)
%yy1=mapminmax('reverse',o2',ps2);%normalise reverse
y1predict=o2';
yy1=[y1';y1predict];
plot(1:length(train),yy1)
legend('real','predict')
ylabel('boston price')
title('trainning set score')
