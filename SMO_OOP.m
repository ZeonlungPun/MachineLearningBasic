%SOM OOP main
clear
x=xlsread('wine.xlsx','A2:M179');
SOM=SOMnetwork(5,0.1,0.5,0.01,30000)
SOM.train(x,3,1)
label=SOM.cluster(x);
y=xlsread('wine.xlsx','N2:N179');
true=3-y;
acc=sum(label==true)./length(label)