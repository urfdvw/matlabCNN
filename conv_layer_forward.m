function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: struct with output data

%% parameter passing from the input data
h_in = input.height; % width of the input
w_in = input.width; % height of the input
c_in = input.channel; % number of channels in the input
batch_size = input.batch_size; % number of batch samples

%% parameter passing from the layer setting
k = layer.k; % width of the filter
pad = layer.pad; % number of padding pixels for the input image
stride = layer.stride; % stride of the convolution
c_out = layer.num; % number of convolution filters to be learnt

%% construct output
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;
assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
% construct
output.batch_size = batch_size;
output.height = h_out;
output.width = w_out;
output.channel = c_out;
output.data = zeros([h_out*w_out*c_out, batch_size]);

%% for each datum in the minibatch, find output
for b = 1 : batch_size
    image_in = reshape(input.data(:, b), [h_in, w_in, c_in]);
    image_out = zeros([h_out, w_out, c_out]);
    % for each output channel
    for c = 1 : c_out
        filter = reshape(param.w(:, c), [k, k]);
        bias = param.b(c);
        %% TODO, filtering
    end
    output.data(:, b) = image_out(:);
end
end

