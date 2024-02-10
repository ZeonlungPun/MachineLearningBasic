%Radical Bias Function neural network 1
%rbf distance
function result1=guass(sigma,x,ci)
dist=sum((x-ci).^2,2);
result1=exp(-dist./(2*sigma.^2));
return
