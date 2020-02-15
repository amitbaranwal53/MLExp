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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

for i = 1:m,
  input = [1 X(i,:)]; #1 * 401  Thata1 = 25*401  Theta2 = 10*26
  layer_2_output_without_sigmoid = input * Theta1'; #1*25
  layer_2_output = sigmoid(layer_2_output_without_sigmoid); #1*25
  layer_2_output = [1 layer_2_output]; #1*26
  layer_3_output = sigmoid(layer_2_output * Theta2'); #1*10
  output = layer_3_output'; #10*1
  actual_output = zeros(size(output)); #10*1
  actual_output(y(i)) = 1;
  res = ((- actual_output) .* log(output)) - ((1 - actual_output) .* log(1 - output));
  J = J + sum(res);

  delta3 = output - actual_output; #10*1
  delta2_pre = Theta2' * delta3; #26*1
  delta2 = delta2_pre(2:end) .* (sigmoidGradient(layer_2_output_without_sigmoid))'; #25*1
  #delta2 = delta2(2:end);#25*1

  Theta2_grad = Theta2_grad + delta3 * layer_2_output; #10*1 - 1*26 : 10*26
  Theta1_grad = Theta1_grad + delta2 * input; #25*1 - 1*401 : 25*401
end

J = J / m;
Theta1_WithoutBias = Theta1(:,2:end);
theta1_sum = sum(sum(Theta1_WithoutBias .* Theta1_WithoutBias))  * (lambda / (2*m));

Theta2_WithoutBias = Theta2(:,2:end);
theta2_sum = sum(sum(Theta2_WithoutBias .* Theta2_WithoutBias)) * (lambda / (2*m));

J = J + theta1_sum + theta2_sum;

reg_term_Theta2_grad = Theta2 * (lambda / m);
reg_term_Theta2_grad = [zeros(num_labels, 1) reg_term_Theta2_grad(:,2:end)];

reg_term_Theta1_grad = Theta1 * (lambda / m);
reg_term_Theta1_grad = [zeros(hidden_layer_size, 1) reg_term_Theta1_grad(:,2:end)];

Theta2_grad = reg_term_Theta2_grad + Theta2_grad / m;
Theta1_grad = reg_term_Theta1_grad + Theta1_grad / m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
