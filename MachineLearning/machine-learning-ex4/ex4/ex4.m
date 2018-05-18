
%% Initialization
clear ; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============


% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

% Randomly select 12 data points to display
sel = randperm(size(X, 1));
sel = sel(1:12);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;
i=1;
j=0;
k=1;
v=1;
Xtrain = zeros(m-1000,400);
ytrain = zeros(m-1000,1);
Xtest = zeros(1000,400);
ytest = zeros(1000,1);
for i=1:500:m
    for j=0:499
        if(j<100)
        Xtest(k,:)=X(i+j,:);
        ytest(k,:)=y(i+j,:);
        k=k+1;
        else
            Xtrain(v,:)=X(i+j,:);
            ytrain(v,:)=y(i+j,:);
            v=v+1;
        end
    end
        
end

pause;
%% ================ Part 6: Initializing Pameters ================


fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(hidden_layer_size,input_layer_size+1);
initial_Theta2 = randInitializeWeights(num_labels, hidden_layer_size+1);

% Unroll parameters




%% =================== Part 8: Training NN ===================

%
fprintf('\nTraining Neural Network... \n')


lambda = 1;
Iterations = 1000;
Alpha = 2;


[J,Theta1,Theta2] = Momentum_Update(Xtrain,ytrain,initial_Theta1,initial_Theta2,lambda, ...
                                  Iterations,Alpha);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 10: Implement Predict =================


pred = predict(Theta1, Theta2, Xtest);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);


%%===================    end  =========================================%%


