function [nll, g, od, percent] = mlrloss(wb, X, y, K, gpu, prediction)
% Implements the loss layer
% input
%   wb: [(N+1)*(K-1)] *1 col vector, a concatenated weights and bias
%   X: input of loss layer
%   y: training labels
%   K: K distinct classes (1 to K), for MNIST it is 10
%   gpu: To use GPU for computation or not, should be 0 beasue not used
%   prediction: 
% output
%   nll: scaler, cost 
%   g: gradients
%   od: derivatives
%   percent: percentage of correct prediction, if prediction is true

%% convert data type for GPU if necessary
% should not be used
if gpu == 1
    X = single(X); y = double(y);
end

%% pass parameters
[N,M] = size(X); % N features M examples
% restore w and b
theta = reshape(wb(1:N*(K-1)), K-1, N);
bias  = reshape(wb((1+N*(K-1)):end), K-1, 1);

%% convert labels y into K*M one-hot label array
% I indexes into the correct target entries
I=full(sparse(y,1:M,1,K,M));

%% soft-max
% Compute the values after the linear transform
W=[ bsxfun(@plus, theta * X, bias) ; zeros(1, M) ];

% This rescales so that all values are negative, hence, no overflow
% problems with the exp operation (a single +inf can blow things up)
W=bsxfun(@minus, W, max(W));
W=exp(W);

% Convert to Probabilities by normalizing
P=bsxfun(@rdivide, W, sum(W));

%% Loss.
% P(logical(I)) is bool indexing
nll=-full(sum(log(P(logical(I)))));
if prediction == 1
    [~, indices] = max(P); % take the index of maximum value
    percent = sum(y-indices== 0) / length(y);    
else
    percent = 0;
end
% Compute the gradients
if (nargout >= 2)
    od = (P - I); % P-I gives exactly the error derivatives at the "output units"
    % after this theta' * od can be used as the backprop derivative
    % while od * X can be used as the derivative at the current layer
    gw = od * X';
    gw = gw(1:K-1,:); 
    gb = sum(od, 2);
    gb = gb(1:K-1,:);
    g = [gw(:) ; gb(:)];
end

% Compute the derivatives for backprop
if (nargout >= 3)
    % use this for backprop
    od = theta' * od(1:K-1,:);
end

end