%knn algorithm
datasets=xlsread('�����Ʋ�.xlsx','A1:E46');
numtestratio=0.25;%���Լ�ռ��
precise1=[];
precise2=[];
m=10;
for i=1:m%��������m��ȡƽ��ֵ
    r=randperm(length(datasets));%�������������������������
    num_test=round(length(datasets)*numtestratio);
    test=datasets(r(1:num_test),:);%���Լ�����
    r(1:num_test)=[];%�ù����������
    train=datasets(r,:);%ѵ����
    precision1=run_knn(train);%ѵ����׼ȷ��k=3:12
    precision2=run_knn(test);%���Լ�׼ȷ��
    precise1=[precise1;precision1];
    precise2=[precise2;precision2];
end
average1=mean(precise1)
average2=mean(precise2)
sigama1=std(precise1,0)
sigama2=std(precise2,0)
plot(3:12,average2)%���Ʋ��Լ�ѧϰ����
hold on
plot(3:12,average2+sigama2)
hold on
plot(3:12,average2-sigama2)
xlabel('K')
ylabel('precison rate')
title('KNN�㷨ѧϰ����')
legend('��ֵ+1����׼��','��ֵ','��ֵ-1����׼��')