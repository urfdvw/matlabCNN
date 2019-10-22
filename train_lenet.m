clear
close all
clc

rand('seed', 100000)
randn('seed', 100000)

%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false; % load a small data set
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% combine training and validate set into a large training set
xtrain = [xtrain, xvalidate]; % combine data
ytrain = [ytrain, yvalidate]; % combine label
m_train = size(xtrain, 2); % get the number of datum of combined set

%% Parameters initalization
% parameters to get learning rate
% lr = base_lr * (1 + gamma * iter) ^ (- power)
epsilon = 0.01;
gamma = 0.0001;
power = 0.75;
w_lr = 1; % base rate for w
b_lr = 2; % base rate for b

% parameters for momentum sdg
mu = 0.9;
weight_decay = 0.0005;

% learning process parameters
batch_size = 100; % minibatch size
test_interval = 500; % display test accuracy for every ? tierations
display_interval = 100; % display training loss for every ? tierations
snapshot = 500; % save trained network parameter for every ? tierations
max_iter = 3000; % number of training iterations
no_epochs = 100;


%% init the parameters
if 1
    % by using the following to train from scratch
    params = init_convnet(layers);
else
    % by loading the network
    load lenet_pretrained.mat
end

% prepare memory of each layer catch for momentum
param_winc = params;
for l_idx = 1:length(layers)-1
    param_winc{l_idx}.w = zeros(size(param_winc{l_idx}.w));
    param_winc{l_idx}.b = zeros(size(param_winc{l_idx}.b));
end

%% Training the network
% shuffle the training set
new_order = randperm(m_train); % random index
xtrain = xtrain(:, new_order);
ytrain = ytrain(:, new_order);

curr_batch = 1; % current batch start index, used for taking minibatch
for iter = 1 : max_iter
    %% take a mini batch
    % if the rest of the data set is not large enough to form a minibatch
    if (curr_batch > m_train)
        % re-shuffle the training set
        new_order = randperm(m_train);
        xtrain = xtrain(:, new_order);
        ytrain = ytrain(:, new_order);
        % reset the start index of the current batch
        curr_batch = 1;
    end
    % take a subset from traing data
    %   which starts from curr_batch
    %   and size is batch_size
    x_batch = xtrain(:, curr_batch:(curr_batch+batch_size-1));
    y_batch = ytrain(:, curr_batch:(curr_batch+batch_size-1));
    curr_batch = curr_batch + batch_size; % update position of curr_batch
    %% forward and backward propogation
    [cp, param_grad] = conv_net(params, layers, x_batch, y_batch);
    %% SGD parameter update
    for l_idx = 1:length(layers)-1
        % We have different epsilons for w and b. Calling get_lr and sgd_momentum twice.
        w_rate = get_lr(iter, epsilon*w_lr, gamma, power); % get learning rate
        [w_params, w_params_winc] = sgd_momentum(w_rate, mu, weight_decay, params, param_winc, param_grad); % momentum sgd update
        
        b_rate = get_lr(iter, epsilon*b_lr, gamma, power); % get learning rate
        [b_params, b_params_winc] = sgd_momentum(b_rate, mu, weight_decay, params, param_winc, param_grad); % momentum sgd update
        
        params{l_idx}.w = w_params{l_idx}.w;
        params_winc{l_idx}.w = w_params_winc{l_idx}.w;
        params{l_idx}.b = b_params{l_idx}.b;
        params_winc{l_idx}.b = b_params_winc{l_idx}.b;
    end
    
    %% display progress
    % disp training loss
    if mod(iter, display_interval) == 0
        fprintf('cost = %f training_percent = %f\n', cp.cost, cp.percent);
    end
    % disp test accuracy
    if mod(iter, test_interval) == 0
        layers{1}.batch_size = size(xtest, 2);
        [cptest] = conv_net(params, layers, xtest, ytest);
        layers{1}.batch_size = batch_size;
        fprintf('test accuracy: %f \n\n', cptest.percent);
    end
    % save trained net
    if mod(iter, snapshot) == 0
        filename = 'lenet.mat';
        save(filename, 'params');
    end
end
