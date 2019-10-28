function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initializing values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
thet=theta;
thet(1)=0;

J=(-1/m)*sum((y.*log(sigmoid(X*theta)))+(1-y).*log(1-sigmoid((X*theta))));
J=J+((lambda/(2*m))*sum(thet.^2));
grad=(1/m)*(X'*(sigmoid(X*theta)-y));
grad=grad+((lambda/m)*thet);


grad = grad(:);

end
