function precision1=run_knn(sample)%��knn��Ӧ�����к���
class1=[];%���շ��ص�Ԥ��ֵ
x_sample=sample(:,1:end-1);
precision1=[];
for k=3:12;%ʵ�鲻ͬ��k
    for i=1:length(x_sample);%��Ԥ�����ݴ��������Խ��
        p=x_sample(i,:);%��������������з������
        predict1=knn_classify(p,sample,k);%��predict���շ�����
        class1=[class1;predict1];
    end
    compare1=(class1==round(sample(:,end)));
    precision=sum(compare1==1)./length(x_sample);%׼ȷ��
    precision1=[precision1,precision];
    class1=[];
end

