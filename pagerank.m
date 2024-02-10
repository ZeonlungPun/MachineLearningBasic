matrix=xlsread('new.xlsx','sheet4','B5:BWT1974');%��������͸�ӱ�
matrix(find(isnan(matrix)==1))=0;%���ȱֵ
m=matrix./sum(matrix);%ת�ƾ���
d=0.75;%����ϵ��
[n,~]=size(m);
r1=(1/n)*ones(n,1);%��ʼ��page rankֵΪ1/n
%page rank �����㷨
while 1
    r2=d.*m*r1+((1-d)/n).*ones(n,1);%������ֵ��Ϊ������
    error=r2-r1;
    if abs(error)<0.01
        break
    end
    r1=r2;
end
%page rank �ݷ�
x0=ones(n,1);%��ʼֵ
a=d*m+((1-d)/n)*ones(n,n);
while 1
    y0=a*x0;
    x1=y0./max(abs(y0));
    if max(abs(x1-x0))<0.01
        break
    end
    x0=x1;
end
R=x1./sum(x1);%���ս��