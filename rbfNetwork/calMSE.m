%Radical Bias Function neural network 4
%calculating sum of squre of error
function sse=calMSE(ypredict,y)
sse=0.5*sum((y-ypredict).^2)./size(y,1);
return