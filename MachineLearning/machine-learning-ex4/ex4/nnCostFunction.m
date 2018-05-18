function [J,Theta1_grad,Theta2_grad] = nnCostFunction(Theta1,Theta2, ...
                                   X, y, lambda)

%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% Setup some useful variables
m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));



X = [ones(m,1),X];
X  = X';
a1 = X;
z2 = Theta1*X;
a2 = sigmoid(z2);
a2 = [ones(1,m);a2];
z3 = Theta2*a2;
a3 = sigmoid(z3);
h = a3;
yi = zeros(size(h));
for i =1:m
    yi(y(i),i)= 1;
end
J1= -yi.*log(h)- (1-yi).*log(1-h) ;
J1=sum(J1(:));
a=J1/m;
J=a;
%--------Regularization-----------%
S = 0;
K= 0;
Theta1r = Theta1.^2;
Theta2r = Theta2.^2;
S = sum(Theta1r(:))-sum(Theta1r(:,1));
K = sum(Theta2r(:))-sum(Theta2r(:,1));
S = S/m;
K = K/m;
R = S+K;
R = lambda*(R/2);
J = a + R;
%------------Back Prop------------%

d3 = h-yi;
Theta21 = Theta2(:,2:end);
d2 = (Theta21'*d3).*sigmoidGradient(z2);

Dell2 = d3*a2'; 
Dell1 = d2*a1';
D2 = Dell2/m;
D1 = Dell1/m;
T1=[zeros(size(Theta1,1),1) Theta1(:,2:end)];
T2=[zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad=D1+((lambda/m)*T1);
Theta2_grad=D2+((lambda/m)*T2);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
