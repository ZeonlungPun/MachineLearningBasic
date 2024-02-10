%���ģ��һЩ����������;
random=unifrnd(-1,1,15,2);
group1=ones(15,2)+random;
group2=2.65*ones(15,2)+random;
x=[group1;group2]
data=[group1,-1*ones(15,1);group2,1*ones(15,1)];%�������Ӧ��ǩ
y=data(:,end)%��ʵ���
%��֪��,�������ݶ��½�����ά
count=0%��������ͳ��
lam=0.1%����
w=ones(size(x,2),1);%��ʼֵ
b=1;
while 1
    count=count+1;
    missing_index=-1;%��ʼ�������������
    for i=1:length(data)
        checking=y(i).*(x(i,:)*w+b);
        if checking <=0%%�ҵ�������
            missing_index=i;
            break%ÿ��ѭ��ֻ��һ��������,�ҵ���forѭ������ִ��
        end
    end
    if missing_index==-1%��û�������㣬��ʾ��֪���ѹ������,����whileѭ��
            break
    end
    w=w+lam*y(missing_index).*x(missing_index,:)';%ÿ��ֻ��һ����������е���
    b=b+lam*y(missing_index);
    
end
%�������Լ���Ӧ�ĳ�ƽ��
scatter(group1(:,1),group1(:,2),'red');
hold on
scatter(group2(:,1),group2(:,2),'blue');
hold on

k=-w(1)./w(2);%��ֱ�߸�д��б��ʽ������ͼ
bb=-b./w(2);
xx=0:4;
yy=k.*xx+bb;
plot(xx,yy,'--')
title('perceptron')
xlabel('dimension1')
ylabel('dimension2')
legend('group1','group2','separating hyperplane')

    

