%Softmax regression
%����������
data=xlsread('wine.xlsx','B2:O179');
xraw=data(:,1:end-1);
x=(xraw-min(xraw))./(max(xraw)-min(xraw));%��һ��
data1=[ones(length(data),1),x,data(:,end)+1];
w00=unifrnd(-1,1,size(xraw,2)+1,1);%�涨��ʼֵ
w01=unifrnd(-1,1,size(xraw,2)+1,1);%�涨��ʼֵ
w02=unifrnd(-1,1,size(xraw,2)+1,1);%�涨��ʼֵ
w02=w00;
w00=w02;
numtestratio=0.3;%���Լ�ռ��
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
per=[]
value=[]
for i =1:length(xtrain)%�����ʼֵ
    pt=exp(xtrain(i,:)*w00)+exp(xtrain(i,:)*w01)+exp(xtrain(i,:)*w02);
    p0=exp(xtrain(i,:)*w00)./pt;
    p1=exp(xtrain(i,:)*w01)./pt;
    p2=exp(xtrain(i,:)*w02)./pt;
    val=[p0,p1,p2];
     
    [value1,index]=max(val')
    per=[per;index-1]%��ǰԤ�����
    value=[value;val];%Ԥ��Ϊ3�����ĸ��ʣ�����-1��ʾ���
end

num=5000%��������
stopvalue=0.001%ֹͣ����
pt1=[];
for hh=1:num%�ݶȵĵ�������
    grad0=[];
    for i=1:length(xtrain);%�ݶȵĵ�һ������
        if per(i)==0;
            pt=1;
        else
            pt=0;
        end
        pt1=[pt1;pt];
        g0=xtrain(i,:).*(pt-value(i,1));
        grad0=[grad0;g0];
    end
    grad0=-sum(grad0)./length(xtrain);
    grad1=[];
    for i=1:length(xtrain)%�ݶȵĵڶ�������
        if per(i)==1%ÿѭ��һ�ζ������һ��
            pt=1;
        else
            pt=0;
        end
        g1=xtrain(i,:).*(pt-value(i,2));
        grad1=[grad1;g1];
    end
    grad1=-sum(grad1)./length(xtrain);
    grad2=[];
    for i=1:length(xtrain)%�ݶȵĵ���������
        if per(i)==2
            pt=1;
        else
            pt=0;
        end
        g2=xtrain(i,:).*(pt-value(i,3));
        grad2=[grad2;g2];
    end
    grad2=-sum(grad2)./length(xtrain);
    
    yita=0.0001;
    w=[w00';w01';w02'];
    grad=[grad0;grad1;grad2];%�ݶȾ���
    w=w-yita.*grad;%����Ȩ�ؾ���
  
    w00=w(1,:)';%���ºõĸ����Ȩ��
    w01=w(2,:)';
    w02=w(3,:)';
    %����ѵ�����������ڸ������ĸ���
    h0=[];
    h1=[];
    h2=[];
    for i=1:length(xtrain);
        h0=[h0;exp(xtrain(i,:)*w00)];
    end
    p0=h0./sum(h0);
    for i=1:length(xtrain);
        h1=[h1;exp(xtrain(i,:)*w01)];
    end
    p1=h1./sum(h1);
    for i=1:length(xtrain);
        h2=[h2;exp(xtrain(i,:)*w02)];
    end
    p2=h2./sum(h2);
    ptrain=[p0,p1,p2];%�������ڸ������ĸ���
    [a1,a2]=max(ptrain');
    per=a2'-1;
end
trainpre=per
trainscore=sum(trainpre==train(:,end))./length(xtrain)

%%������Լ��������ڸ������ĸ���
h0=[];
h1=[];
h2=[];
for i=1:length(xtest);
    h0=[h0;exp(xtest(i,:)*w00)];
end
p0=h0./sum(h0);
for i=1:length(xtest);
    h1=[h1;exp(xtest(i,:)*w01)];
end
p1=h1./sum(h1);
for i=1:length(xtest);
    h2=[h2;exp(xtest(i,:)*w02)];
end
p2=h2./sum(h2);
ptest=[p0,p1,p2];%�������ڸ������ĸ���
[a1,a2]=max(ptest');
testpre=a2-1;
testscore=sum(testpre'==test(:,end))./length(xtest)

