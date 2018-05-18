function [J,Theta1, Theta2] = Momentum_Update( X,y,initial_Theta1,initial_Theta2,lambda, ...
    Iterations, Alpha )
%Nestrov Momentum Update
Theta1 = initial_Theta1;

Theta2 = initial_Theta2;
i=1;
mu = 0.9;
v1=zeros(size(Theta1));
v2=zeros(size(Theta2));
Alpha = 0.1;
while (i<1000)
    Theta1 = Theta1 + mu*v1;
    Theta2 = Theta2 + mu*v2;
    [J(i),Theta1_grad,Theta2_grad] = nnCostFunction(Theta1,Theta2,X, y, lambda);
    v1 = mu*v1 - Alpha*Theta1_grad;
    v2 = mu*v2 - Alpha*Theta2_grad;
    Theta1 = Theta1 +v1 ;
    Theta2 = Theta2 +v2;
    disp('The cost function is');
    disp(J(i));
    i= i+1;
    
end

end

