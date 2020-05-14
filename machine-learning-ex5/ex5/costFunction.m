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
XTHETA = X*theta;  % theta'X, but since we want to compute all the rows at once, we do it the other way around
HX = sigmoid(XTHETA); % our hypothesis value of the sigmoid function

%% COST CALCULATION
Y1 = -y .* log(HX); % our cost function if y=1
Y0 = (1 .- y) .* log( 1 .- HX ); % our cost if y=0
LOGCOST = Y1 - Y0; % logistic regression costs of all the rows (data set)
J = sum(LOGCOST) / m; % average of all logistic regression cost;

%% GRADIENT CALCULATION, intuition lies on the calculus (partial derivative)
LINCOST = HX - y; % just subtracting our hypothesis value with the real value, not a usable cost in logistic regression;
MULT = LINCOST .* X; % multiply all in X by LINCOST
SUMS = sum(MULT); % sum all the columns
AVGS = SUMS / m; % average all the columns
grad = AVGS; % and we got our new theta for the next iteration of the gradient descent

% =============================================================

end
