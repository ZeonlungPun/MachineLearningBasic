%logistic regression 
%multinational classification
data=xlsread('wine1.xlsx','B2:R179');

xraw=data(:,1:end-4);
x=(xraw-min(xraw))./(max(xraw)-min(xraw));%��һ��
y1=data(:,end-3);
y2=data(:,end-2);
y3=data(:,end-1);
data1=[ones(length(data),1),x,y1,y2,y3,data(:,end)];
w01=ones(size(xraw,2)+1,1);%�涨��ʼֵ
w02=ones(size(xraw,2)+1,1);%�涨��ʼֵ
w03=ones(size(xraw,2)+1,1);%�涨��ʼֵ
numtestratio=0.25;%���Լ�ռ��
r=randperm(length(data));%�������������������������
num_test=round(length(data)*numtestratio);
test=data1(r(1:num_test),:);%���Լ�����
r(1:num_test)=[];%�ù����������
train=data1(r,:);%ѵ����
xtrain=train(:,1:end-4);
ytrain=train(:,end-3:end-1);
xtest=test(:,1:end-4);
ytest=test(:,end-3:end-1);
yita=0.02%����
stop_value=0.001%����ֵֹͣ
num=20000%��������
grad=[];
for i=1:num
    gradient1=zeros(size(xraw,2)+1,1);
    gradient2=zeros(size(xraw,2)+1,1);
    gradient3=zeros(size(xraw,2)+1,1);
    for j=1:length(xtrain);
        p1=1./(1+exp(-xtrain(j,:)*w01));
        p2=1./(1+exp(-xtrain(j,:)*w02));
        p3=1./(1+exp(-xtrain(j,:)*w03));
        grad1=(ytrain(j,1)-p1)*xtrain(j,:)';
        grad2=(ytrain(j,2)-p2)*xtrain(j,:)';
        grad3=(ytrain(j,3)-p3)*xtrain(j,:)';
        gradient1=gradient1+grad1;%����ÿһ�ε��ݶȣ������ݶ��½���
        gradient2=gradient2+grad2;
        gradient3=gradient3+grad3;
    end
        grad=[grad;gradient1,gradient2,gradient3];
        w01=w01+yita*gradient1;%�����ݶ�
        w02=w02+yita*gradient2;
        w03=w03+yita*gradient3;
     if (gradient1'*gradient1).^0.5<stop_value
         break
     end
end
p11=1./(1+exp(-xtrain*w01));
p12=1./(1+exp(-xtrain*w02));
p13=1./(1+exp(-xtrain*w03));
p21=1./(1+exp(-xtest*w01));
p22=1./(1+exp(-xtest*w02));
p23=1./(1+exp(-xtest*w03));
ptrain=[p11,p12,p13]';
ptest=[p21,p22,p23]';
[~,b2]=max(ptrain);
[~,c2]=max(ptest);
ytrainperdict=[b2'];
ytestperdict=[c2'];
ytrianreal=train(:,end);
ytestreal=test(:,end);
yy1=(ytrianreal==(ytrainperdict-1));
yy2=(ytestreal==(ytestperdict-1));
trainscore=sum(yy1)./length(ytrainperdict)%ѵ��������
testscore=sum(yy2)./length(ytestperdict)%���Լ�����