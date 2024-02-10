%dynamic programming 维特比算法，泛化版本 %HMM隐马尔可夫模型
function [road,p]=viterbi(o,a,b,pi)%o:观测序列,a:状态转移矩阵；b:状态序列;pi:初始状态概率序列
T=length(o);%获取总时间
[N,M]=size(b);%获取状态数量N和观测数量M
yita1=pi.*b(:,o(1));%计算t=1的概率最大值
for t=2:T%计算t=2至其它
    eval(['yita',num2str(t),'=[]']);
    eval(['fi',num2str(t),'=[]']);
    for i=1:N
        [number,index]=max(eval(['yita',num2str(t-1),'']).*a(:,i));
        eval(['yita',num2str(t),'(i,1)=number.*b(i,o(t))'])
        eval(['fi',num2str(t),'(i,1)=index']);
    end
end
eval(['[p,i',num2str(t),']=max(yita',num2str(t),')']);%求解最优路径概率和最优路径的终点
for t=1:T-1%求解其它路径
    tt=T-t;
    eval(['i',num2str(tt),'=fi',num2str(tt+1),'(i',num2str(tt+1),');'])
end
road=[];%添加所有路径
t=1
while t<=T
     eval(['road=[road,i',num2str(t),']'])
     t=t+1;
end
return
