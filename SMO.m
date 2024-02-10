%SMO4 train
clear;
x=xlsread('wine.xlsx','A2:M179');
x=mapminmax(x',0,1);%对输入变量归一化
x=x';
[samples,features]=size(x);
%初始化各节点位置以及各个节点之间的位置
m=3;n=1;
gridlocation=init_grid(m,n);
griddist=calGdist(gridlocation);
%初始化各节点对应权值
w=poslin(randn(m*n,features));
steps=5*samples;
for i=1:steps
    %竞争，随机选取样本计算距离
    no=randperm(samples);
    data=x(no(1),:);
    xdist=[];
    for j=1:size(w,1)
        dd=pdist2(data,w(j,:));
        xdist=[xdist;dd];
    end
    %找优胜点
    [~,winnerID]=min(xdist);
    %迭代，确定学习率和节点优胜半径，并保存
    max_rate=0.5;min_rate=0.01;
    max_diameter=5;min_diameter=0.1;
    [rate,diameter]=changerate(i,steps,max_rate,min_rate,max_diameter,min_diameter);
    %圈定优胜领域所有节点
    judge=(griddist(winnerID,:)<diameter);
    winnerroundID=find(judge~=0);
    w(winnerroundID,:)=  w(winnerroundID,:)+rate*(data-w(winnerroundID,:));
end
label=cluster(x,w);
y=xlsread('wine.xlsx','N2:N179');
true=3-y;
acc=sum(label==true)./length(label)
    