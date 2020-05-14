function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
oneX = [ones(m, 1) X]; % adding bias feature > 5000x401
z2 = oneX*Theta1'; % > layer 2 computation > 5000x401 * 401*25 = 5000x25
gz2 = sigmoid(z2); % layer 2 sigmoid squishification > 5000x25

gz2 = [ones(m,1) gz2]; % adding bias unit on layer 2 > 5000x26
z3 = gz2*Theta2'; % layer 3 unit computation > 5000x26 * 26x10 = 5000x10
gz3 =sigmoid(z3); % layer 3 squishification > 5000x10

loghx = log(gz3); % 5000 x 10
minloghx = log(1.-gz3); % 5000 x 10

yy = [1:num_labels] == y; % 5000 x 10

postcost = -yy.*loghx; % 5000 x 10
negcost = (1-yy).*minloghx; % 5000 x 10

simplecost = postcost - negcost;  % 5000 x 10
sums = sum(simplecost(:)); % Converting the matrix to a column vector and then summing the resultant column vector
J = sums/m;

Theta1sq = Theta1(:,2:end).^2;
Theta2sq = Theta2(:,2:end).^2;

Theta1sum = sum(Theta1sq(:));
Theta2sum = sum(Theta2sq(:));

regularization = (Theta1sum+Theta2sum)*(lambda/(2*m));

J = J+regularization;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%% There is actually a correct calculus based intuition to all the operation that's needed for backprop below
%% But the caption I put down is not according to the calculus based intuition
%% Nevertheless, the caption still serve as a good starting point intuition to understand backprop
%% To understand the calculus based intuition, one must google: 'backprop calculus' to truly understand what does backprop do
%% It has something to do with measuring how tiny nudges in each weights affect the cost of the network
%% The most crucial computation on backprop are based on calculus chain rule on the above measurement

%STEP1
a1 = oneX; % X including bias term
a2 = gz2; % layer 2 computation: 5000x25
a3 = gz3; % layer 3 or output computation: 5000 x 10

%STEP2
del3 = a3 - yy; % error costs of layer 3: 5000 x 10

%STEP3
theta2tdel3 = del3 * Theta2; % 'disperse' or 'spread' error of layer 3 to all unit on layer two relative to their weight: 5000x10 10x26
sigmgrad2 = sigmoidGradient([ones(m,1) z2]); % gradient of layer 2 sigmoid, used to add higher cost to irrelevant result, and lower cost to relevant result: 5000x26
del2 = theta2tdel3 .* sigmgrad2; % error costs of layer 2: multiply the dispersal value with the gradient weighting factor: 5000x26

%STEP4
del2nobias = del2(:,2:end); % removing the bias of del2: 5000x25

delta2 = del3'*a2; % 10x5000 5000x26 = 10x26
delta1 = del2nobias'*a1; % 25*5000 5000x400 = 25x401

%STEP5
Theta2_grad = delta2/m;
Theta1_grad = delta1/m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
