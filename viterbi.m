%dynamic programming ά�ر��㷨�������汾 %HMM������ɷ�ģ��
function [road,p]=viterbi(o,a,b,pi)%o:�۲�����,a:״̬ת�ƾ���b:״̬����;pi:��ʼ״̬��������
T=length(o);%��ȡ��ʱ��
[N,M]=size(b);%��ȡ״̬����N�͹۲�����M
yita1=pi.*b(:,o(1));%����t=1�ĸ������ֵ
for t=2:T%����t=2������
    eval(['yita',num2str(t),'=[]']);
    eval(['fi',num2str(t),'=[]']);
    for i=1:N
        [number,index]=max(eval(['yita',num2str(t-1),'']).*a(:,i));
        eval(['yita',num2str(t),'(i,1)=number.*b(i,o(t))'])
        eval(['fi',num2str(t),'(i,1)=index']);
    end
end
eval(['[p,i',num2str(t),']=max(yita',num2str(t),')']);%�������·�����ʺ�����·�����յ�
for t=1:T-1%�������·��
    tt=T-t;
    eval(['i',num2str(tt),'=fi',num2str(tt+1),'(i',num2str(tt+1),');'])
end
road=[];%�������·��
t=1
while t<=T
     eval(['road=[road,i',num2str(t),']'])
     t=t+1;
end
return
