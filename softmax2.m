data=xlsread('wine.xlsx','B2:O179');
xraw=data(:,1:end-1);
x=(xraw-min(xraw))./(max(xraw)-min(xraw));%��һ��
data1=[ones(length(data),1),x,data(:,end)+1];
w01=unifrnd(-1,1,size(xraw,2)+1,1);%�涨��ʼֵ
w02=unifrnd(-1,1,size(xraw,2)+1,1);%�涨��ʼֵ
w03=unifrnd(-1,1,size(xraw,2)+1,1);%�涨��ʼֵ
%w01=ones(size(xraw,2)+1,1);
%w02=ones(size(xraw,2)+1,1);
%w03=ones(size(xraw,2)+1,1);
w=[w01';w02';w03'];%��ʼȨ�ؾ���
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
%׼������

per=[];
value=[];
for i =1:length(xtrain)%�����ʼֵ
    pt=exp(-xtrain(i,:)*w01)+exp(-xtrain(i,:)*w02+exp(-xtrain(i,:)*w03));
    p1=exp(-xtrain(i,:)*w01)./pt;
    p2=exp(-xtrain(i,:)*w02)./pt;
    p3=exp(-xtrain(i,:)*w03)./pt;
    val=[p1,p2,p3];
    [value1,index]=max(val');
    per=[per;index];%��ǰԤ�����
    value=[value;val];%Ԥ��Ϊ3�����ĸ��ʣ�������ʾ���
end
y=zeros(length(xtrain),3);
for i=1:length(xtrain);
    label=per(i);
    y(i,label)=1;%��ʼ���������
end

yita=0.01;%����
num=50000;%��������
for j=1:num
    grad=-((y-value)'*xtrain)./length(data);%�ݶȵ���
    w=w+yita*grad;%Ȩ�ص���
    w01=w(1,:)';%���ºõĸ����Ȩ��
    w02=w(2,:)';
    w03=w(3,:)';
    value=[];
    per=[];
    for i=1:length(xtrain);%ѭ������ÿһ���������ڸ����ĸ���
        pt=exp(-xtrain(i,:)*w01)+exp(-xtrain(i,:)*w02+exp(-xtrain(i,:)*w03));
        p1=exp(-xtrain(i,:)*w01)./pt;
        p2=exp(-xtrain(i,:)*w02)./pt;
        p3=exp(-xtrain(i,:)*w03)./pt;
        val=[p1,p2,p3];%ÿһ���������ڸ����ĸ���
        [value1,index]=max(val');
        per=[per;index];%��ǰԤ�����
        value=[value;val];%Ԥ��Ϊ3�����ĸ��ʣ�������ʾ���
    end
    y=zeros(length(xtrain),3);
    for i=1:length(xtrain);
        label=per(i);
        y(i,label)=1;%���������
    end
end
trainpre=per;
trainscore=sum(trainpre==train(:,end))./length(xtrain)
value=[];
per=[]
 for i =1:length(xtest)%���Լ����
    pt=exp(-xtest(i,:)*w01)+exp(-xtest(i,:)*w02+exp(-xtest(i,:)*w03));
    p1=exp(-xtest(i,:)*w01)./pt;
    p2=exp(-xtest(i,:)*w02)./pt;
    p3=exp(-xtest(i,:)*w03)./pt;
    val=[p1,p2,p3];
    [value1,index]=max(val');
    per=[per;index];%��ǰԤ�����
    value=[value;val];%Ԥ��Ϊ3�����ĸ��ʣ�������ʾ���
 end
testpre=per;
testscore=sum(testpre==test(:,end))./length(xtest)

