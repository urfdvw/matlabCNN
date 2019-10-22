function [xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset)
    % This function is used to generate random data set for training
    % inputs:
    %   fullset: bool, generate large set if true, small sets if false
    % outputs:
    %   xtrain: [0~1] double, training data, 784 rows, each col is a datum
    %   ytrain: [1,2, ... , 10], col vector, labels for training data
    %   xvalidate: [0~1] double,calidate data, 784 rows, each col is a datum
    %   yvalidate: [1,2, ... , 10], col vector, labels for validate data
    %   xtest: [0~1] double, test data, 784 rows, each col is a datum
    %   ytest: [1,2, ... , 10], col vector, labels for test data

    %% read matlab MNIST data file
	load('mnist_all.mat');
    % each datum is a 1*784 vector
    % each data matrix is a N*784 vector that has N datum
    
    %% combine data
    % concatenate all training data into a big matrix
	xtrain = [train0; train1; train2;train3;train4;train5;train6;train7;train8;train9];
	% creat labels for the training data, so that data '0' have label 1,
	% '9' have label '10'
    ytrain = [ones(size(train0,1),1);
    	2*ones(size(train1,1),1);
    	3*ones(size(train2,1),1);
    	4*ones(size(train3,1),1);
    	5*ones(size(train4,1),1);
    	6*ones(size(train5,1),1);
    	7*ones(size(train6,1),1);
    	8*ones(size(train7,1),1);
    	9*ones(size(train8,1),1);
    	10*ones(size(train9,1),1)];
    % concatenate all test data
	xtest = [test0; test1; test2;test3;test4;test5;test6;test7;test8;test9];
    % label all test data
	ytest = [ones(size(test0,1),1);
    	2*ones(size(test1,1),1);
    	3*ones(size(test2,1),1);
    	4*ones(size(test3,1),1);
    	5*ones(size(test4,1),1);
    	6*ones(size(test5,1),1);
    	7*ones(size(test6,1),1);
    	8*ones(size(test7,1),1);
    	9*ones(size(test8,1),1);
    	10*ones(size(test9,1),1)];
    
    %% convert image data into double formate range [0,1] 
	xtrain = double(xtrain)/255;
	xtest = double(xtest)/255;
    %% shuffle data sets
	p = randperm(size(xtrain, 1)); % random index for data and label
	xtrain = xtrain(p, :);
	ytrain = ytrain(p, :);
	p = randperm(size(xtest, 1));
	xtest = xtest(p, :);
	ytest = ytest(p, :);
    
    %% seperate training data set into real 'training' data and validate data
	m_validate = 10000; % size of validation set
	xvalidate = xtrain(1:m_validate, :); % validation data
	yvalidate = ytrain(1:m_validate, :); % validation labels
	xtrain = xtrain(m_validate + 1:end, :); % real training set
	ytrain = ytrain(m_validate + 1:end, :); % real training labels
	m_train = size(xtrain, 1); % size of real traing data
	m_test = size(xtest, 1); % size of test data
    
    %% transpose
    % make each data colume a data and each row a feature of the data
	xtrain = xtrain';
	ytrain = ytrain';
	xvalidate = xvalidate';
	yvalidate = yvalidate';
	xtest = xtest';
	ytest = ytest';
    
    %% if not full set
	if ~fullset
        % reduce the amount of data for training, validate and test
		m_train_small = m_train/20;
		m_test_small = m_test/20;
		m_validate_small = m_validate/20;
        % take sub set of the data sets
		xtrain = xtrain(:, 1:m_train_small);
		ytrain = ytrain(:, 1:m_train_small);
		xtest = xtest(:, 1:m_test_small);
		ytest = ytest(:, 1:m_test_small);
		xvalidate = xvalidate(:, 1:m_validate_small);
		yvalidate = yvalidate(:, 1:m_validate_small);
	end
end
