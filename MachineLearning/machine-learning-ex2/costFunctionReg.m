function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
n=size(X);
k=0;p=0;t=0;r=0;
l=X*theta;
for i=1:m
k =k + [-y(i)*log(sigmoid(l(i))) - (1-y(i))*log(1-sigmoid(l(i)))] ;
end
for j=2:n(2)
 t=t+theta(j).^2;
end
J=k*(1/m)+ t*(lambda/(2*m));

for i=1:n(2)
    p=0;
    for j=1:m
 p= p+(sigmoid(l(j))-y(j))*X(j,i);
    end
    if (i==1)
  grad(i) = p*(1/m);
    else
grad(i) = p*(1/m)+ theta(i)*(lambda/(m));
     end
end


% =============================================================

end
