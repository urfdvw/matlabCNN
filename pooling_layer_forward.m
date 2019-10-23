function [output] = pooling_layer_forward(input, layer)
%% parameter passing from the input data
h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;

%% parameter passing from the layer setting
k = layer.k;
pad = layer.pad;
stride = layer.stride;

%% construct output
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;
assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
% construct
output.height = h_out;
output.width = w_out;
output.channel = c;
output.batch_size = batch_size;
output.data = zeros([h_out*w_out*c, batch_size]);

%% for each datum in the minibatch, find output
for b = 1 : batch_size
    image_in = reshape(input.data(:, b), [h_in, w_in, c_in]);
    image_out = zeros([h_out, w_out, c_out]);
    % for each output channel
    for c = 1 : c_out
        image_channel = zeros([h_out, w_out]);
        %% TODO, pooling
        for h = 1 : h_out
            for w = 1 : w_out
                image_channel(h, w) = 
            end
        end
        
        %%
        image_out(:, :, c) = image_channel;
    end
    output.data(:, b) = image_out(:);
end
end

