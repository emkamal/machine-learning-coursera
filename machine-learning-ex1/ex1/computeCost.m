function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

HX = X*theta; % predicted value based on the current theta
COST = HX-y; % predicted value subtracted by the actual value
COSTSQUARE = COST .^ 2; % Cost squared so that it will all be positive, and give larger cost for irrelevant prediction
AVERAGE = sum(COSTSQUARE)/m; % average of cost
J = AVERAGE / 2; % divide by 2 cos https://math.stackexchange.com/questions/884887/why-divide-by-2m

% =========================================================================

end
