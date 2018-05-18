function W = randInitializeWeights(L_out, L_in)

ep = 0.12;
W = rand(L_out,L_in)*(2*ep)-ep;

% Next Time do Xavier Initialization

% =========================================================================

end
