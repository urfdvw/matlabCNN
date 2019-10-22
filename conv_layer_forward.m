function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: struct with output data

%% 
% parameter passing
h_in = input.height; % width of the input
w_in = input.width; % height of the input
c = input.channel; % number of channels in the input
batch_size = input.batch_size;
k = layer.k; % width of the filter
pad = layer.pad; % number of padding pixels for the input image
stride = layer.stride; % stride of the convolution
num = layer.num; % number of convolution filters to be learnt
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

% throw an error if the size of input, filter, padding and stride do not
% match
assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')

% ?
input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

%% Fill in the code
% Iterate over the each image in the batch, compute response,
% Fill in the output datastructure with data, and the shape. 


end

