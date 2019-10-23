function [output] = relu_forward(input)

%% relu layers do not change the data size
output.height = input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;

%% ReLU operation
output.data = zeros(size(input.data));
% TODO
output.data = input.data;
output.data(output.data < 0) = 0;
end
