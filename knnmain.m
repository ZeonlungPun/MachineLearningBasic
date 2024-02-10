%knn algorithm
datasets=xlsread('银行破产.xlsx','A1:E46');
numtestratio=0.25;%测试集占比
precise1=[];
precise2=[];
m=10;
for i=1:m%程序运行m次取平均值
    r=randperm(length(datasets));%根据样本数量生成随机数索引
    num_test=round(length(datasets)*numtestratio);
    test=datasets(r(1:num_test),:);%测试集数据
    r(1:num_test)=[];%用过的索引清空
    train=datasets(r,:);%训练集
    precision1=run_knn(train);%训练集准确率k=3:12
    precision2=run_knn(test);%测试集准确率
    precise1=[precise1;precision1];
    precise2=[precise2;precision2];
end
average1=mean(precise1)
average2=mean(precise2)
sigama1=std(precise1,0)
sigama2=std(precise2,0)
plot(3:12,average2)%绘制测试集学习曲线
hold on
plot(3:12,average2+sigama2)
hold on
plot(3:12,average2-sigama2)
xlabel('K')
ylabel('precison rate')
title('KNN算法学习曲线')
legend('均值+1倍标准差','均值','均值-1倍标准差')