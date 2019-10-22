function [output, P] = convnet_forward(params, layers, data)
% Forward propogation of the  net work, but without loss layer
% input:
%   params: cell array of structures, parameters of each layer
%   layers: cell array of structures, network defination of each layer
%   data: batch of data without label
% output:
%   output: cell array of structures representing the output data of each
%   layer
%   P (optional): soft max output of the lastlayer

%% acquire the number of layers
l = length(layers);

%% build the output of the first layer from scratch 
% in/output structure attrbutes
%   data: array, number of index depends on layer tpoe, value of the in/output
%   height: scaler, height of a image;
%   width: scaler, width of a image;
%   channel: scaler, number of channels;
%   batch_size: scaler number of images in a batch of image;
%   diff: derivatives for backprop

% make sure the first layer is data
assert(strcmp(layers{1}.type, 'DATA') == 1, 'first layer must be data layer');
output{1}.data = data;
output{1}.height = layers{1}.height;
output{1}.width = layers{1}.width;
output{1}.channel = layers{1}.channel;
output{1}.batch_size = layers{1}.batch_size;
output{1}.diff = 0;
%% loop through the second layer to the second last to get result
for i = 2:l-1
    % choose one forward type according to layer setting
    switch layers{i}.type
        case 'CONV'
            output{i} = conv_layer_forward(output{i-1}, layers{i}, params{i-1});
        case 'POOLING'
            output{i} = pooling_layer_forward(output{i-1}, layers{i});
        case 'IP'
            output{i} = inner_product_forward(output{i-1}, layers{i}, params{i-1});
        case 'RELU'
            output{i} = relu_forward(output{i-1});
        case 'ELU'
            output{i} = elu_forward(output{i-1}, layers{i});
    end
end

%% compute soft max as P
% this part is only used for prediction
if nargout > 1
    W = bsxfun(@plus, params{l-1}.w * output{l-1}.data, params{l-1}.b);
    W = [W; zeros(1, size(W, 2))];
    W=bsxfun(@minus, W, max(W));
    W=exp(W);
    
    % Convert to Probabilities by normalizing
    P=bsxfun(@rdivide, W, sum(W));
end
end