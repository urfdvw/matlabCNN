function params = init_convnet(layers)
% Initialise the network weights
% inputs:
%   layers: cell of structure, layer{i} is the structure defination of
%           layer i
% outputs:
%   params: cell of structure, parameters of the network
%           params{i}.b: biases of layer i
%           params{i}.w: weights of layer i
            

% get the size of the first layer
h = layers{1}.height;
w = layers{1}.width;
c = layers{1}.channel;
% loop through all layers
for i = 2:length(layers)
    switch layers{i}.type
        case 'CONV'
            % random set the parameters by eq ?
            scale = sqrt(3/(h*w*c));
            params{i-1}.w = 2*scale*rand(layers{i}.k*layers{i}.k*c/layers{i}.group, layers{i}.num) - scale;
            params{i-1}.b = zeros(1, layers{i}.num);
            % get the output size of this layer
            h = (h + 2*layers{i}.pad - layers{i}.k) / layers{i}.stride + 1;
            w = (w + 2*layers{i}.pad - layers{i}.k) / layers{i}.stride + 1;
            c = layers{i}.num;
        case 'POOLING'
            % pooling layer does not have parameters
            params{i-1}.w = [];
            params{i-1}.b = [];
            % get the output size of this layer
            h = (h - layers{i}.k) / layers{i}.stride + 1;
            w = (w - layers{i}.k) / layers{i}.stride + 1;
        case 'IP' % fully connected layer
            switch layers{i}.init_type
                case 'gaussian'
                    % random set the parameters by Gaussian Random Variables eq ?
                    scale = sqrt(3/(h*w*c));
                    params{i-1}.w = scale*randn(h*w*c, layers{i}.num);
                    params{i-1}.b = zeros(1, layers{i}.num);
                case 'uniform'
                    % random set the parameters by Uniform Random Variables eq ?
                    scale = sqrt(3/(h*w*c));
                    params{i-1}.w = 2*scale*rand(h*w*c, layers{i}.num) - scale;
                    params{i-1}.b = zeros(1, layers{i}.num);
            end
            % get the output size of this layer
            h = 1;
            w = 1;
            c = layers{i}.num;
        case 'RELU'
            % ReLU layer does not have parameters
            params{i-1}.w = [];
            params{i-1}.b = [];
            % output size is the same as input
        case 'ELU'
            % eLU layer does not have parameters
            params{i-1}.w = [];
            params{i-1}.b = [];
            % output size is the same as input
        case 'LOSS'
            % random set the parameters by eq ?
            scale = sqrt(3/(h*w*c));
            % last layer is K-1
            params{i-1}.w = 2*scale*rand(h*w*c, layers{i}.num - 1) - scale;
            params{i-1}.w = params{i-1}.w';
            params{i-1}.b = zeros(1, layers{i}.num - 1);
            params{i-1}.b = params{i-1}.b';
            % get the output size of this layer
            h = 1;
            w = 1;
            c = layers{i}.num;
    end
end
end
