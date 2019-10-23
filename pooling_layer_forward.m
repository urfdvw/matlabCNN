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
    image_in = reshape(input.data(:, b), [h_in, w_in, c]);
    %% TODO add padding to imput image
    image_in_pad = zeros([h_in + 2 * pad, w_in + 2 * pad, c]);
    image_in_pad(pad+1:pad+h_in, pad+1:pad+w_in, :) = image_in;
    
    image_out = zeros([h_out, w_out, c]);
    %% TODO, max pooling
    for ic = 1 : c
        for h = 1 : h_out
            for w = 1 : w_out
                y = 0;
                for ki = 1 : k
                    for kj = 1 : k
                        y = max(y, image_in_pad(stride*(h-1) + (ki-1) + 1, stride*(w-1) + (kj-1) + 1, ic));
                    end
                end
                image_out(h, w, ic) = y;
            end
        end
    end
    output.data(:, b) = image_out(:);
end
end

