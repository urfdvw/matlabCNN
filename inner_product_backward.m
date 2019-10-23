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

% % Replace the following lines with your implementation.
% param_grad.b = zeros(size(param.b)); % you need to remove this line 
% param_grad.w = zeros(size(param.w)); % you need to remove this line 
% 
% 
% output.diff; % d_out by batch size
% input.data; % d_in by batch size
% param.w; % d_in by d_out
% input_od; % d_in by batch size
% param_grad.w; % d_in by d_out
% param_grad.b; % 1 by d_out

input_od = param.w * output.diff;
param_grad.w = input.data * output.diff';
param_grad.b = mean(output.diff,2)';
end
