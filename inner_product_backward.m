function [param_grad, input_od] = inner_product_backward(output, input, layer, param)
% backward step for the IP (fully connected) layer
% inputs:
%   output
%   input
%   layer
%   param
% outputs:
%   param_grad
%   input_od

% Replace the following lines with your implementation.
param_grad.b = zeros(size(param.b)); % you need to remove this line 
param_grad.w = zeros(size(param.w)); % you need to remove this line 

end
