%GAUSSIAN process
rng(0,'twister');
N = 1000;
x = linspace(-10,10,N)';
y = sin(x) + 0.01*randn(N,1);

[m,n]=size(x);
index=randperm(m);
ratio=0.9;
num=round(ratio*m);
id1=index(1:num);
id2=index(num+1:end);
[trainx,trainy]=deal(x(id1,:),y(id1,:));
[testx,testy]=deal(x(id2,:),y(id2,:));

gram1=trainx'*trainx;
h1=repmat(diag(gram1),size(trainx,1),1);
dist1=h1+h1'-2*gram1;

gram2=testx'*testx;
h2=repmat(diag(gram2),size(testx,1),1);
dist2=h2+h2'-2*gram2;

%kernel function parameters
sigma=0.1;
k1=exp(-dist1/(2*sigma^2));
k2=exp(-dist2/(2*sigma^2));

dist12=[];
for i=1:size(trainx,1)
    for j=1:size(testx,1)
        dist12(i,j)=(trainx(i,:)-testx(j,:)).^2;
    end
end
k12=exp(-dist12/(2*sigma^2));

mu_new=k12'*inv(k1+0.01*eye(size(k1,1)))*(trainy-mean(trainx'))+mean(testx');

scatter(1:length(testy),testy,'r')
hold on
plot(1:length(testy),zscore(mu_new),'blue')
%performance of testdata
X=[ones(length(testy),1),testy];
[~,~,~,~,states]=regress(zscore(mu_new),X);
r2=states(1)

error=zscore(mu_new)-testy;
stem(error)


