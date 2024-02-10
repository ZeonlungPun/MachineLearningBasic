%%Radical Bias Function neural network 6
%predicting and scoring 
 
h_output=change(sigma1,x,c);%hidden layer output(m,h)
yi_input=addintercept(h_output);%output layer input(m,h+1)
predicting=yi_input*w;%output layer predict(m,1)
mse=calMSE(yi_output,y)
