function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

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

n = length(theta);
x = sigmoid(X * theta);

sum = 0;

for i = 1:m,
  if y(i) == 0,
    sum = sum - log(1 - x(i));
  elseif y(i) == 1,
    sum = sum - log(x(i));
  end;
end;

sum2 = 0;
for j = 2:n,
  sum2 = sum2 + theta(j) * theta(j);
end;

J = sum/m + (lambda * sum2)/(2 * m);

grad = (X' * (x - y))/m;

for j = 2:n,
  grad(j) = grad(j) + lambda * theta(j) / m;
end;

% =============================================================

end
