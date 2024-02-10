%K-Nearest Neighbours algorithm;二分类
function class=knn_classify(p,train,k)
%p需要判别类型的未知点；datasets数据集；k选取的最近的k个点;
x=train(:,1:end-1);%提取数据
n1=length(train);
y=train(:,end);%提取标签
y=round(y);%将标签固定为整数
xt=[x;p];
xt=(xt-max(xt))./(max(xt)-min(xt));%将判别点和原始数据一起作归一化处理
xx=xt(1:n1,:);%提取出归一化后的原始数据
p=xt(n1+1:end,:);%提出出归一化后的待判别点
dsqure=(sum(((xx-p).^2)'))';%计算距离
new=[dsqure,y];%新矩阵，用于存储计算结果
new1=sortrows(new,1);%第一列升序排列，其他列跟着动
result=new1(1:k,2);%提取出距离最近的k个点
zeronum=sum(result==0);
onenum=sum(result==1);
class=[];%储存分类判别结果
if zeronum>=onenum
    class=0
elseif onenum>zeronum
    class=1
end
return 




 

