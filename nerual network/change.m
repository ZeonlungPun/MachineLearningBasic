%Radical Bias Function neural network 2
%transforming the raw datas into new data
function newx=change(sigma,x,c)
newx=[];
for i=1:size(c,1)
    center=c(i,:);
    newx(:,i)=guass(sigma(i),x,c(i,:));
end
return