%֧��������support vector machine%����soft margin
%���ι滮����
%���ģ��һЩ����������;
random=unifrnd(-1,1,50,2);
group1=ones(50,2)+random;
group2=2.85*ones(50,2)+random;
x=[group1;group2];
data=[group1,-1*ones(50,1);group2,1*ones(50,1)];%�������Ӧ��ǩ
y=data(:,end);%��ʵ���
h=[]%Ŀ�꺯����H
for i=1:length(x)%��������������Ҫ����
    for j=1:length(x)
        h(i,j)=x(i,:)*(x(j,:))'*y(i)*y(j);
    end
end
f=-1*ones(length(x),1);%Ŀ�꺯����f
%����Լ��
aeq=y';
beq=zeros(1,1);
%����ʽԼ��
ub=[];
ib=[];
%�Ա���Լ��
lb=zeros(length(x),1);
ub=[];
[a,fval]=quadprog(h,f,ib,ub,aeq,beq,lb,ub);%���ι滮����
w=0;%ϵ������
u=0
for i=1:length(a)
    if a(i)<1e-05
        a(i)=0;
    end
end
ff=find(a~=0)
j=ff(1)%Ѱ��aϵ�������ڵ��±�j
for i=1:length(x)%�ؼ�ϵ�������
    w=w+a(i)*y(i)*x(i,:)';
    u=u+a(i)*y(i)*(x(i,:)*x(j,:)');
end
b=y(j)-u;
%�������Լ���Ӧ�ĳ�ƽ��
scatter(group1(:,1),group1(:,2),'red');
hold on
scatter(group2(:,1),group2(:,2),'blue');
hold on
k=-w(1)./w(2);%��ֱ�߸�д��б��ʽ������ͼ
bb=-b./w(2);
xx=0:4;
yy=k.*xx+bb;
plot(xx,yy,'-')
hold on
yy=k.*xx+bb+1./w(2)
plot(xx,yy,'--')
hold on
yy=k.*xx+bb-1./w(2)
plot(xx,yy,'--')
title('support vector machine')
xlabel('dimension1')
ylabel('dimension2')
legend('group1','group2','separating hyperplane')
%������ߺ�����Ԥ��
predict=[];
for i=1:length(x)%Ԥ���i������
    uu=0;%���ȱ���
    for j=1:length(x)%����ѵ������������������Ԥ�⺯��
        uu=uu+a(j)*y(j)*(x(j,:)*x(i,:)');
    end
    result=sign(uu+b);
    predict(i,1)=result;
end
judge=(predict==y);
score=sum(judge)./length(data)