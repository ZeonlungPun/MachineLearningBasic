matrix=xlsread('new.xlsx','sheet4','B5:BWT1974');%读入数据透视表
matrix(find(isnan(matrix)==1))=0;%填补空缺值
m=matrix./sum(matrix);%转移矩阵
d=0.75;%阻尼系数
[n,~]=size(m);
r1=(1/n)*ones(n,1);%初始化page rank值为1/n
%page rank 迭代算法
while 1
    r2=d.*m*r1+((1-d)/n).*ones(n,1);%最后迭代值即为输出结果
    error=r2-r1;
    if abs(error)<0.01
        break
    end
    r1=r2;
end
%page rank 幂法
x0=ones(n,1);%初始值
a=d*m+((1-d)/n)*ones(n,n);
while 1
    y0=a*x0;
    x1=y0./max(abs(y0));
    if max(abs(x1-x0))<0.01
        break
    end
    x0=x1;
end
R=x1./sum(x1);%最终结果