%K-Nearest Neighbours algorithm;������
function class=knn_classify(p,train,k)
%p��Ҫ�б����͵�δ֪�㣻datasets���ݼ���kѡȡ�������k����;
x=train(:,1:end-1);%��ȡ����
n1=length(train);
y=train(:,end);%��ȡ��ǩ
y=round(y);%����ǩ�̶�Ϊ����
xt=[x;p];
xt=(xt-max(xt))./(max(xt)-min(xt));%���б���ԭʼ����һ������һ������
xx=xt(1:n1,:);%��ȡ����һ�����ԭʼ����
p=xt(n1+1:end,:);%�������һ����Ĵ��б��
dsqure=(sum(((xx-p).^2)'))';%�������
new=[dsqure,y];%�¾������ڴ洢������
new1=sortrows(new,1);%��һ���������У������и��Ŷ�
result=new1(1:k,2);%��ȡ�����������k����
zeronum=sum(result==0);
onenum=sum(result==1);
class=[];%��������б���
if zeronum>=onenum
    class=0
elseif onenum>zeronum
    class=1
end
return 




 

