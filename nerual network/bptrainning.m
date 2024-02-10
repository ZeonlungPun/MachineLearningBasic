%back propagation neural network2
%trainning function with forward and backward propagation
function [w_hidden1,w_hidden2,w_xinput,errorlist]=bptrainning(x1,y1,r,iters,tolerance,w_xinput,w_hidden1,w_hidden2)
%r:learnning rate iters:Maximum Iterations
errorlist=[];%error changing with iteration
for i=1:iters
    %forward propagation
    [yi_output,hidden_input1,hidden_input2]=calresult(w_xinput,w_hidden1,w_hidden2,x1,size(x1,1));
    mse=calMSE(y1,yi_output);
    errorlist=[errorlist;mse];
    if mse< tolerance
        break
    end
    %backward propagation and changing weight
    [w_hidden1,w_hidden2,w_xinput]=bwpropagation(w_xinput,w_hidden1,w_hidden2,y1,yi_output,r,hidden_input1,hidden_input2,x1);
  
end
return
    