%back propagation neural network4
%backward propagation

function [w_hidden1,w_hidden2,w_xinput]=bwpropagation(w_xinput,w_hidden1,w_hidden2,y1,ypred,r,hidden_input1,hidden_input2,x1)
bias=ones(length(y1),1);
%changing weight of output layer to hidden layer2
error=ypred-y1;
grad_output=error.*logsig(ypred).*(1-logsig(ypred));
hi2=[bias,hidden_input2];
w0=w_hidden2;
w_hidden2=w_hidden2-r*hi2'*grad_output;
%%changing weight of hidden layer2 to hidden layer1
grad_hidden2=(grad_output*w0(2:end)').*logsig(hidden_input2).*(1-logsig(hidden_input2));
hi=[bias,hidden_input1];
w_hidden1=w_hidden1-r*hi'*grad_hidden2;
%changing weight of hidden layer1 to input layer
grad_hidden1=(grad_output*w0(2:end)').*logsig(hidden_input1).*(1-logsig(hidden_input1));
xi=[bias,x1];
w_xinput=w_xinput-r*xi'*grad_hidden1;
return