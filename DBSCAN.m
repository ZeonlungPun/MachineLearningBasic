%DBSCAN algorithm
%�ܶȾ��෨
%d�������ݼ�
%minpts���������ٵ���
%esp�뾶
data=xlsread('dbscandata.xlsx','B2:D201');
d=data(:,1:2);
minpts=15
esp=0.3
dist=squareform(pdist(d,'euclidean'))%�������,��Ҫ�������þ������������ĺ���
judge1=(dist<=esp)
[k1,k2]=find((judge1==1)&(sum(judge1,2)>=minpts))%�Һ��ĵ�
core=unique(k1)%������ĵ�����
%��ʼ�����-1����δ����
label=-1*ones(length(d),1);
cluster=1;
%�������к��ĵ�
for i=1:length(core)
    j=core(i);
    if  label(j)==-1;%������ĵ�δ�����࣬������Ϊ�����ӵ㣬��ʼѰ����Ӧ�ؼ�
        label(j)=cluster;%���Ƚ����ʶΪ�Ѳ�������ǰ���
        %Ѱ��λ������������ڵġ�δ������ĵ㣬����������Ӽ���
        neighbour=find((dist(:,j)<=esp)&(label==-1));%��¼�����ӵ��������������Ӽ���
        seed=neighbour;
 %ͨ�����ӵ㣬��ʼ������Ѱ���ܶȿɴ�����ݵ㣬һֱ�����Ӽ���Ϊ�գ�һ���ؼ�Ѱ�����
        while ~isempty(seed)
            sign=randperm(length(seed));%����һ�������ӵ����
            new=seed(sign(1))%���ӵ����
            label(new)=cluster;%���µ���Ϊ��ǰ��
            seed(sign(1))=[];%ɾ����seed�����ж�Ӧλ���ϵĴ�������������
    %Ѱ��newPoint���ӵ�eps���򣨰����Լ���
            results=find(dist(:,new)<=esp);
 %���newPoint���ں��ĵ㣬��ônewPoint�ǿ�����չ�ģ����ܶ��ǿ���ͨ��newPoint�����ܶȿɴ��
            if length(results)>=minpts%����������û�б�����ĵ�ѹ�����Ӽ���
                for r=1:length(results);
                    jj=results(r);
                    if label(jj)==-1;
                        seed=[seed;jj];
                    end
                end
            end
        end
        cluster=cluster+1;%�ؼ�������ϣ�Ѱ�ҵ�һ�����    
    end   
end
%ԭʼЧ��ͼ 
d1=data(data(:,3)==0,1:2)
d2=data(data(:,3)==1,1:2)
scatter(d1(:,1),d1(:,2),'red')
hold on
scatter(d2(:,1),d2(:,2),'blue')
hold on
%�������ĵ���Χ��Բ
for i=1:length(core)
    j=core(i);%��ȡ���ĵ�����
    xx=d(j,1);%������
    yy=d(j,2);%������
    r=esp%�뾶
    theta = 0:pi/20:2*pi;    %�Ƕ�[0,2*pi] 
    x = xx+r*cos(theta);
    y = yy+r*sin(theta);
    plot(x,y,'-')
    axis equal
    hold on
end


            
            
        