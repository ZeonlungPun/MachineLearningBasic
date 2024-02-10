%logistics regresion algorithm
data1=xlsread('breast_cancer','A2:AF570');%��ȡȫ������
x=data1(:,1:end-1)%��ȡ�Ա���
%xx=(x-min(x))./(max(x)-min(x))%��һ��[0,1]
xx=x;
data=[ones(length(data1),1),xx,data1(:,end)];
numtestratio=0.3;%���Լ�ռ��
r=randperm(length(data));%�������������������������
num_test=round(length(data)*numtestratio);
test=data(r(1:num_test),:);%���Լ�����
r(1:num_test)=[];%�ù����������
train=data(r,:);%ѵ����
xtrain=train(:,1:end-1)
ytrain=train(:,end)
xtest=test(:,1:end-1)
ytest=test(:,end);
yita=0.002%����
stop_value=0.001%����ֵֹͣ
num=8000%��������
w=[];%��¼Ȩ�ر仯
w0=ones(size(data(:,1:end-1),2),1)%�涨��ʼֵ
%batch gardient descent
for i=1:num
    gradient=zeros(size(data1,2),1);%��ʱ������ÿһ�ζ�Ҫ���¼���
    for j=1:length(xtrain);
        p=1./(1+exp(-xtrain(j,:)*w0));
        grad=(ytrain(j,1)-p)*xtrain(j,:)';
        gradient=gradient+grad;%����ÿһ�ε��ݶȣ������ݶ��½���
    end
        w0=w0+yita*gradient;%�����ݶ�
        w=[w,w0];%Ϊ�˿��ӻ�ÿһ��W�ĵ������
     if (gradient'*gradient).^0.5<stop_value
         break
     end
end
p1=1./(1+exp(-xtrain*w0));
p2=1./(1+exp(-xtest*w0));
ratio=tabulate(data1(:,end));
p0=ratio(end,3)./100%��ֵ���趨
ytrainperdict=( p1>p0);
ytestperdict=(p2>p0)
yy1=(ytrain==ytrainperdict);
yy2=(ytest==ytestperdict);
trainscore=sum(yy1)./length(ytrainperdict)%ѵ��������
testscore=sum(yy2)./length(ytestperdict)%���Լ�����
plot(1:size(w,2),w)%ѧϰ����1����Ȩ�����ŵ������������ӵı仯���
title('ѧϰ����1����Ȩ�����ŵ������������ӵı仯���')

