%DBSCAN algorithm
%密度聚类法
%d输入数据集
%minpts包含的最少点数
%esp半径
data=xlsread('dbscandata.xlsx','B2:D201');
d=data(:,1:2);
minpts=15
esp=0.3
dist=squareform(pdist(d,'euclidean'))%距离矩阵,需要巧妙利用距离矩阵所代表的含义
judge1=(dist<=esp)
[k1,k2]=find((judge1==1)&(sum(judge1,2)>=minpts))%找核心点
core=unique(k1)%储存核心点索引
%初始化类别，-1代表未分类
label=-1*ones(length(d),1);
cluster=1;
%遍历所有核心点
for i=1:length(core)
    j=core(i);
    if  label(j)==-1;%如果核心点未被分类，将其作为的种子点，开始寻找相应簇集
        label(j)=cluster;%首先将点标识为已操作（当前类别）
        %寻找位于这个点邻域内的、未被分类的点，将其放入种子集合
        neighbour=find((dist(:,j)<=esp)&(label==-1));%记录下种子的索引，放入种子集合
        seed=neighbour;
 %通过种子点，开始生长，寻找密度可达的数据点，一直到种子集合为空，一个簇集寻找完毕
        while ~isempty(seed)
            sign=randperm(length(seed));%弹出一个新种子点操作
            new=seed(sign(1))%种子点序号
            label(new)=cluster;%将新点标记为当前类
            seed(sign(1))=[];%删除在seed矩阵中对应位置上的处理过的种子序号
    %寻找newPoint种子点eps邻域（包含自己）
            results=find(dist(:,new)<=esp);
 %如果newPoint属于核心点，那么newPoint是可以扩展的，即密度是可以通过newPoint继续密度可达的
            if length(results)>=minpts%将邻域内且没有被分类的点压入种子集合
                for r=1:length(results);
                    jj=results(r);
                    if label(jj)==-1;
                        seed=[seed;jj];
                    end
                end
            end
        end
        cluster=cluster+1;%簇集生长完毕，寻找到一个类别    
    end   
end
%原始效果图 
d1=data(data(:,3)==0,1:2)
d2=data(data(:,3)==1,1:2)
scatter(d1(:,1),d1(:,2),'red')
hold on
scatter(d2(:,1),d2(:,2),'blue')
hold on
%画出核心点周围的圆
for i=1:length(core)
    j=core(i);%提取核心点索引
    xx=d(j,1);%横坐标
    yy=d(j,2);%纵坐标
    r=esp%半径
    theta = 0:pi/20:2*pi;    %角度[0,2*pi] 
    x = xx+r*cos(theta);
    y = yy+r*sin(theta);
    plot(x,y,'-')
    axis equal
    hold on
end


            
            
        