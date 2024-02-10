rng(123)
temp = xlsread('139jw1206.xlsx', 'a');
temp1 = xlsread('139jw1206.xlsx', 'b');
 data = [temp; temp1];
 %save data.mat data
%load data
%W = 1./[10 0.08334 10 10 10 0.0833	0.125 0.125	0.1875 0.03125 0.03125];
%W=[5/77,1.5/11,5/77,1.5/11,5/77,5/77,1.5/11,1.5/11,5/77,5/77,5/77];
W=[10,20,20,50,50,10,30,30,20,20,20];
Newdata = [data(:,3:4), data(:,8:end)];
NewWeight = [W(:,2:3), W(:,7:end)];
%NewWeight = W;
stype_data = [2, 5, 6, 7];
for i=1:length(stype_data)
    tem = stype_data(i);
    featureTemp = data(:,tem);
    setTemp = unique(featureTemp);
    for j=1:length(setTemp)
        featureTemp(featureTemp==setTemp(j)) = j;
        %NewWeight = [NewWeight W(i)];
        NewWeight = [NewWeight 1];
    end
    temp = ind2vec(featureTemp');
    temp = temp';
    temp = full(temp);
    Newdata = [Newdata temp];
end
% 
% save Newdata.mat Newdata 
%load Newdata.mat
 figure;
 geoshow(Newdata(:,4),Newdata(:,3),'DisplayType','point');
 %Newdata = Newdata(1:10:end,:);
data=Newdata;
[data,rule1] = mapminmax(data');
data = data';
NewWeight = repmat(NewWeight, size(data,1), 1);
data =data.*NewWeight;


rng(1)
K=110;
[IDX, C] = kmeans(data, K, 'Distance','sqeuclidean','Replicates',5, 'MaxIter',500);

%[IDX, C] = kmedoids(data,K,'Distance','sqeuclidean');

figure;
colorSet = [];
for i=1:K
    colorSet{i} = [rand() rand() rand()];
 end
for i=1:K
   indice = find(IDX==i);
   tempXY = Newdata(indice,:);
   plot(tempXY(:,3),tempXY(:,4),'.','color',colorSet{i})
   hold on;
end
xlabel('经度')
ylabel('纬度')
title('中国陆地景观特征分区图(K-means）')
grid on;
figure;
for i=1:K
    xIndice = fix(i/10);
    yIndice = mod(i,10);
    if yIndice==0
        xIndice = xIndice-1;
        yIndice = 10;
    end
    plot(yIndice*2,38-xIndice*2,'s','MarkerFaceColor',colorSet{i},'markersize',10)
    strText = sprintf('%d',i);
    text(yIndice*2+0.32,38-xIndice*2,strText)
    hold on;
end
axis([-2,24,-4,40])
