%导入数据
rng(2)
option=2;
switch option
    case 1
        data=xlsread('粗化.xlsx','A2:H9167');
    case 2
        data=xlsread('蠕变.xlsx','A2:H9167');
end
%归一化
data_norm=(data-min(data))./(max(data)-min(data));

%k-means
t1=cputime;
K=[];
cost=[];
for k=1:20
    K=[K,k];
    [~,~,sumD,~] = kmeans(data_norm,k,'Replicates',5,'start','cluster','MaxIter',2000);
    sse=sum(sumD);
    cost=[cost,sse];
end
figure(1)
plot(K,cost,'b-','linewidth',2)
hold on
scatter(K,cost,'MarkerFaceColor','blue')
hold on
scatter(4,cost(4),'MarkerFaceColor','k')
xlabel('K')
ylabel('SSE')
title('kmeans clustering:')
t2=cputime;
tt1=t2-t1;
%kmedoids
t1=cputime;
K=[];
cost=[];
for k=1:20
    K=[K,k];
    opts = statset('MaxIter',2000);
    [~,~,sumD,~] = kmedoids(data_norm,k,'Replicates',5,'start','cluster','options',opts);
    sse=sum(sumD);
    cost=[cost,sse];
end
figure(2)
plot(K,cost,'b-','linewidth',2)
hold on
scatter(K,cost,'MarkerFaceColor','blue')
hold on
scatter(4,cost(4),'MarkerFaceColor','k')
xlabel('K')
ylabel('SSE')
title('kmedoids clustering:')
t2=cputime;
tt2=t2-t1;
%FCM
t1=cputime;
K=[];
cost=[];
for k=1:20
    K=[K,k];
    [c,u,obj]=fcm(data_norm,k);
    [~,idx]=max(u',[],2);
    sse=[];
    for j=1:k
        index=find(idx==j);
        xx=data_norm(index,:);
        mu=mean(xx);
        sse1=sum(sum((xx-mu).^2,2));
        sse=[sse,sse1];
    end
    cost=[cost,sum(sse)];
end
figure(3)
plot(K,cost,'b-','linewidth',2)
hold on
scatter(K,cost,'MarkerFaceColor','blue')
hold on
scatter(4,cost(4),'MarkerFaceColor','k')
xlabel('K')
ylabel('SSE')
title('fuzzy C-mean clustering:')
t2=cputime;
tt3=t2-t1;

%Hierarchical cluster
%计算距离矩阵
t1=cputime;
gram=data_norm*data_norm';
h=repmat((diag(gram))',size(data_norm,1),1);
dist=(h+h'-2*gram).^0.5;
% 确定层次聚类树 
treeCluster = linkage(dist,'average'); 
% 可视化聚类树
% dendrogram(treeCluster);
K=[];
cost=[];
for k=1:20
    K=[K,k];
    idx = cluster(treeCluster,'maxclust',k);%划分成k类
    sse=[];
    for j=1:k
        index=find(idx==j);
        xx=data_norm(index,:);
        mu=mean(xx);
        sse1=sum(sum((xx-mu).^2,2));
        sse=[sse,sse1];
    end
    cost=[cost,sum(sse)];
end
figure(4)
plot(K,cost,'b-','linewidth',2)
hold on
scatter(K,cost,'MarkerFaceColor','blue')
hold on
scatter(4,cost(4),'MarkerFaceColor','k')
xlabel('K')
ylabel('SSE')
title('Hierarchical clustering:')
t2=cputime;
tt4=t2-t1;

%SOM聚类
K=[];
cost=[];
for k=1:20
    K=[K,k];
    net=newsom(minmax(data_norm'),[1 k]);
    net.trainparam.epochs=50;
    net=train(net,data_norm');
    y=sim(net,data_norm');
    yy=vec2ind(y);
    idx=yy';
    sse=[];
    for j=1:k
        index=find(idx==j);
        xx=data_norm(index,:);
        mu=mean(xx);
        sse1=sum(sum((xx-mu).^2,2));
        sse=[sse,sse1];
    end
    cost=[cost,sum(sse)];
end
figure(5)
plot(K,cost,'b-','linewidth',2)
hold on
scatter(K,cost,'MarkerFaceColor','blue')
hold on
scatter(4,cost(4),'MarkerFaceColor','k')
xlabel('K')
ylabel('SSE')

