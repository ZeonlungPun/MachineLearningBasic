%%Radical Bias Function neural network 4
%l2 norm
function norm=l2norm(x,c)
[m,~]=size(x);
norm=[];
for i=1:size(c,1)
    norm(:,i)=sum((x-c(i)).^2,2);
end
return