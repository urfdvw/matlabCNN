# Dependency structure

Training 
- train_lenet
    - conv_net
        - convnet_forward
            - conv_layer_forward
            - elu_forward
            - inner_product_forward
            - pooling_layer_forward
            - relu_forward
        - mlrloss
        - conv_layer_backward
        - elu_backward
        - inner_product_backward
        - pooling_layer_backward
        - relu_backward
    - get_lenet
    - get_lr
    - init_convnet
    - load_mnist
    - sgd_momentum

convertion functions
- col2im_conv
    - col2im_conv_matlab
- im2col_conv
    - im2col_conv_matlab

show result
- vis_data
- train_lenet

# definitions of the structure

## params:
- params.w: w\*h\*c by layer{i}.num
- params.b: 1 by layer{i}.num

for convolution layers, w is a 3-index tensor, layer{i}.num is the number of output channels. b means each layer has just 1 bais.
for fully connected layers, in and out puts are considered as vectors, so w in input dimension by output dimension, and number of output channels are output dimensions.
for loss layer, it is the same as fully connected layers except the number of out put channels is reduced by 1.

## in/output
- data: w\*h\*c by batch size
- height: scaler, height of a image;
- width: scaler, width of a image;
- channel: scaler, number of channels;
- batch_size: scaler number of images in a batch of image;
- diff: derivatives for backprop

basically the structure stores the data, data's dimension for later use, and differenciate.
each data[:, i] is actually a 3-index tendor.

## layers
for differnc kinds of layers, the setting parameters are different. but please don't be affraid.