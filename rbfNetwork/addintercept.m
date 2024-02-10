%Radical Bias Function neural network 3
function x1=addintercept(x)
x1=[ones(size(x,1),1),x];
return