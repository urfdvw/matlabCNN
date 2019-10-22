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