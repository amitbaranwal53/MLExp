function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

#size(theta)
#size(X)
#size(y)
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

pred = X * theta;
diff = pred - y;
J = (sum(diff .* diff))/(2*m);

theta_new = theta(2:end, :);
theta_sq_sum = (lambda * (sum(theta_new .* theta_new)))/(2*m);

J = J + theta_sq_sum;

#size(diff)

grad = (X' * diff)/m;
grad_new = (lambda/m) * theta;
grad_new(1) = 0;

grad = grad + grad_new;
% =========================================================================

grad = grad(:);

end
