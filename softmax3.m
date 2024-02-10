clear
data=xlsread('iris.xlsx','A2:E151');
xraw=data(:,1:end-1);
x=(xraw-min(xraw))./(max(xraw)-min(xraw));%��һ��
data1=[ones(length(data),1),x,data(:,end)+1];
%w1=unifrnd(-15,15,size(xraw,2)+1,3);%�涨��ʼֵ
w1=ones(size(xraw,2)+1,3);
w=w1';
%w01=ones(size(xraw,2)+1,1);
%w02=ones(size(xraw,2)+1,1);
%w03=ones(size(xraw,2)+1,1);

numtestratio=0.25;%���Լ�ռ��
r=randperm(length(data));%�������������������������
num_test=round(length(data)*numtestratio);
test=data1(r(1:num_test),:);%���Լ�����
r(1:num_test)=[];%�ù����������
train=data1(r,:);%ѵ����
xtrain=train(:,1:end-1);
ytrain=train(:,end);
xtest=test(:,1:end-1);
ytest=test(:,end);
%׼������:�����ʼֵ
%�þ���˷�����ʽ���㣬��Լʱ��
w01=w(1,:)';%���ºõĸ����Ȩ��
w02=w(2,:)';
w03=w(3,:)';
pt=exp(xtrain*w01)+exp(xtrain*w02)+exp(xtrain*w03);
p1=exp(xtrain*w01)./pt;
p2=exp(xtrain*w02)./pt;
p3=exp(xtrain*w03)./pt;
value=[p1,p2,p3];
[value1,index]=max(value');
per=index';%��ǰԤ�����
y=zeros(length(xtrain),3);
for i=1:length(xtrain);
    label=per(i);
    y(i,label)=1;%��ʼ���������
end
yita=0.01;%����
num=100000;%��������
stopvalue=0.00001;%ֹͣ��������
ww=[];


for j=1:num
    %С�����ݶ��½�
    sample=[xtrain,value,y];
    rr=randperm(length(sample));
    sample_extract=sample(rr(1:round(0.25*length(sample))),:);%ֻ��ȡ0.25��ѵ�������ݶ��½�
    xx=sample_extract(:,1:end-6);
    yy=sample_extract(:,end-2:end);
    vv=sample_extract(:,end-5:end-3);
    grad=-((yy-vv)'*xx)./length(data);%�ݶȵ���
    w=w+yita*grad;%Ȩ�ص���
    w01=w(1,:)';%���ºõĸ����Ȩ��
    w02=w(2,:)';
    w03=w(3,:)';
    ww=[w01';ww];
    %while w00*w00'<stopvalue%�ﵽ���������ֹͣ
     %   break
    %end
    %�µĸ�������
    pt=exp(xtrain*w01)+exp(xtrain*w02+exp(xtrain*w03));
    p1=exp(xtrain*w01)./pt; 
    p2=exp(xtrain*w02)./pt;
    p3=exp(xtrain*w03)./pt;
    value=[p1,p2,p3];
    [value1,index]=max(value');
    per=index';%��ǰԤ�����
    y=zeros(length(xtrain),3);
    for i=1:length(xtrain);
        label=per(i);
        y(i,label)=1;%���������
    end
end
trainpre=per;
trainscore=sum(trainpre==train(:,end))./length(xtrain)%ѵ��������

%���Լ����
pt=exp(xtest*w01)+exp(xtest*w02+exp(xtest*w03));
p1=exp(xtest*w01)./pt;
p2=exp(xtest*w02)./pt;
p3=exp(xtest*w03)./pt;
value=[p1,p2,p3];%Ԥ��Ϊ3�����ĸ��ʣ�������ʾ���
[value1,index]=max(value');
per=index';%��ǰԤ�����
testpre=per;
testscore=sum(testpre==test(:,end))./length(xtest)%���Լ�����
plot(1:num,ww)