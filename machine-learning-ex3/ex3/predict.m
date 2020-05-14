function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % 5000 x 1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X]; % adding bias feature > 5000x401
z2 = X*Theta1'; % > layer 2 computation > 5000x401 * 401*25 = 5000x25
gz2 = sigmoid(z2); % layer 2 sigmoid squishification > 5000x25

gz2 = [ones(m,1) gz2]; % adding bias unit on layer 2 > 5000x26
z3 = gz2*Theta2'; % layer 3 unit computation > 5000x26 * 26x10 = 5000x10
gz3 =sigmoid(z3); % layer 3 squishification > 5000x10

[maxv, p] = max(gz3, [], 2); % getting the column with highest probability on layer 3







% =========================================================================


end
