function [output] = inner_product_forward(input, layer, param)
%% parameter passing from the input data
h_in = input.height;
w_in = input.width;
c_in = input.channel;
d_in = h_in * w_in * c_in; % dimension of input vector
batch_size = input.batch_size;

%% parameter passing from the layer setting
d_out = layer.num; % dimension of output vector

%% construct output
% resolve output shape
h_out = 1;
w_out = 1;
c_out = d_out;
% construct
output.batch_size = batch_size;
output.height = h_out;
output.width = w_out;
output.channel = c_out;
output.data = zeros([d_out, batch_size]);

%% for each datum in the minibatch, find output
for b = 1 : batch_size
    image_in = input.data(:, b); % d_in by 1
    image_out = zeros([d_out, 1]); % d_out by 1
    weights = param.w; % d_in by d_out
    bias = param.b; % 1 by dout
    %% TODO, matrix product
    image_out = (image_in' * weights + bias)';
    %% combine result
    output.data(:, b) = image_out;
end
end
