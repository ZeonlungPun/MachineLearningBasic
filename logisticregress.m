% Logistic regression with gradient descent
a1=a(:,1:5)%�Ա���
a1=zscore(a1)
y=a(:,6)%��ʵ����
x=[ones(40,1),a1]%���ݾ���,ע������1����Ҫ����ƫ��
w=ones(6,1)%��ʼֵ
alpha=0.05%ѧϰ����
grad=0
while abs(grad)>0.000001
    t=x*w
    yh=1./(1+exp(-t))
    error=y-yh
    grad=(x'*error)./40%�ݶȱ��ʽ
    w=w+alpha.*grad
end
for i=1:40
    if (1./(1+exp(-x(i,:)*w)))>0.5
        y1(i,1)=1
    else
        y1(i,1)=0
    end
end
b=(y1==y)
b1=sum(b==1)
b2=b1./40%Ԥ�����ȷ��