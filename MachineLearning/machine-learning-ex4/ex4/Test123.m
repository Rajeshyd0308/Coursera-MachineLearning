
load('ex4data1.mat');
m = size(Xtest, 1);

% You need to return the following variables correctly 
p = zeros(size(Xtest, 1), 1);


h1 = sigmoid([ones(m, 1) Xtest] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);
