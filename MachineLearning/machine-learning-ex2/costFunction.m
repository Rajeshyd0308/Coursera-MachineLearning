function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
n=size(X);
k=0;p=0;
l=X*theta;
for i=1:m
k =k + [-y(i)*log(sigmoid(l(i))) - (1-y(i))*log(1-sigmoid(l(i)))] ;
end
J=k*(1/m);

for i=1:n(2)
    p=0;
    for j=1:m
 p= p+(sigmoid(l(j))-y(j))*X(j,i);
    end
grad(i) = p*(1/m);
end




% =============================================================

end
