function precision1=run_knn(sample)%与knn对应的运行函数
class1=[];%接收返回的预测值
x_sample=sample(:,1:end-1);
precision1=[];
for k=3:12;%实验不同的k
    for i=1:length(x_sample);%将预测数据代入检验测试结果
        p=x_sample(i,:);%逐个样本点代入进行分类测试
        predict1=knn_classify(p,sample,k);%用predict接收分类结果
        class1=[class1;predict1];
    end
    compare1=(class1==round(sample(:,end)));
    precision=sum(compare1==1)./length(x_sample);%准确率
    precision1=[precision1,precision];
    class1=[];
end

