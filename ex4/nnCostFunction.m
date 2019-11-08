function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network.
% 
%   The returned parameter grad is an unrolled vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
Delta1=0;
Delta2=0;   
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%Forward Propagation
   X=[ones(m,1) X];
   a1=X';
   z2=Theta1*a1;
   a2=sigmoid(z2);
   a2=[ones(1,m);a2];
    z3=Theta2*a2;
    a3=sigmoid(z3);
    r=zeros(num_labels,m);
    for i=1:m
     r(y(i),i)=1;
    endfor
      J=(-1/m)*sum(sum((r.*log(a3))+((1-r).*log(1-a3))));
      

Thet1=Theta1(:,2:size(Theta1,2));
Thet2=Theta2(:,2:size(Theta2,2));
reg=lambda/(2*m)*(sum(sum((Thet1.^2)))+sum(sum((Thet2.^2))));
J=J+reg;
%Backpropagation
De3=a3-r;
De2=(Theta2'*De3).*[ones(1,m);sigmoidGradient(z2)];
Delt2=De2(2:size(De2,1),:);
Delta1=Delt2*(a1)';
Delta2=De3*a2';
The1=Theta1;
The1(:,1)=0;
The2=Theta2;
The2(:,1)=0;
%Gradients
Theta1_grad=1/m*Delta1+lambda/m*The1;
Theta2_grad=1/m*Delta2+lambda/m*The2;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
