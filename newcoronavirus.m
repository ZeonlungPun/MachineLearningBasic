%�¹ڷ���������ϣ���ɫVerhulstģ�ͣ�
a=xlsread('newcoronavirus.xlsx','A1:A70')
x1=a'
n=length(x1);
date=0:n-1;
x0=diff(x1);
x0=[x1(1),x0];
for i=2:40
    z1(i)=0.5*(x1(i)+x1(i-1));
end
z1;
B=[-z1(2:40)',z1(2:40)'.^2];
Y=x0(2:40)'
u=inv(B'*B)*B'*Y
b1=u(1);b2=u(2);
for k=0:69
x2(k+1)=(b1.*x0(1))./(b2.*x0(1)+(b1-b2.*x0(1)).*exp(b1.*k))
end
 scatter(date',a)
hold on
plot(date',x2)
xlabel('ʱ��/��','fontsize',12);
ylabel('ȷ�ﲡ����', 'fontsize',12);
legend('ʵ������','Ԥ������');
title('�й��¹ڷ�����ȷ���������������','fontsize',12);
