%back propagation neural network5
%prediction
function y_output=bppredicting(x1,w_xinput,w_hidden1,w_hidden2)
bias=ones(length(x1),1);
xi=[bias,x1];
hidden_input=logsig(xi*w_xinput);%input layer to hidden layer1
hi1=[bias,hidden_input];
hidden_input2=logsig(hi1*w_hidden1);%hidden layer1 to hidden layer2
hi2=[bias,hidden_input2];%hidden layer2 to outputlayer
y_output=logsig(hi2*w_hidden2);
return
