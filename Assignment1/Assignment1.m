function Assignment1()
    % Load data
    [X_train, Y_train, y_train] = LoadBatch("data_batch_1.mat");
    [X_val, Y_val, y_val] = LoadBatch("data_batch_2.mat");
    [X_test, Y_test, y_test] = LoadBatch("test_batch.mat");
    
    % Preprocess data
    mean_X = mean(X_train, 2);  % d x 1
    std_X = std(X_train, 0, 2); % d x 1

    X_train = NormalizeData(X_train, mean_X, std_X);
    X_val = NormalizeData(X_val, mean_X, std_X);
    X_test = NormalizeData(X_test, mean_X, std_X);

    % Initialize parameters
    K = size(Y_train, 1);
    d = size(X_train, 1);
    W = 0.01 * randn(K, d); % Gaussian random values for W
    b = 0.01 * randn(K, 1);  % Gaussian random values for b

    
    P = EvaluateClassifier(X_train(:, 1:100), W, b);
    size(P)
    sum(P, 1)
end 

function ret = NormalizeData(X, mean, std)
    ret = X - repmat(mean, [1, size(X, 2)]);
    ret = ret ./ repmat(std, [1, size(ret, 2)]);
end

function [X, Y, y] = LoadBatch(filename)
    % X contains the image pixel data of size d x n of type double
    % n is the number of images (10'000) and d is the dimensionality of each image (3072 = 32 x 32 x 2)

    % Y is K x n where k is the number of labels (10) and is one-hot encoded of the image label for each image

    % y is a vector of length n containing the label for each image (1-10)

    A = load(filename);
    X = im2double(A.data');
    y = A.labels + 1;
    Y = y == 1:max(y);
    Y = Y';
end

function P = EvaluateClassifier(X, W, b)
    % X = d x n
    % W = K x d
    % b = K x 1
    % P = K x n
    s = W * X + b;
    P = softmax(s);
end

function J = ComputeCost(X, Y, W, b, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % Y = K x n
    % J = scalar
    loss = 1 / size(X,1) * sum(-Y' * log(EvaluateClassifier(X, W, b)));
    
    regularizationTerm = lambda * sum(W.^2, 'all');

    J = loss + regularizationTerm;
end

function acc = ComputeAccuracy(X, y, W, b)
    % X = d x n
    % W = K x d
    % b = K x 1
    % y = n x 1
    % acc = scalar

    P = EvaluateClassifier(X, W, B);
    [~, argmax] = max(P);

    acc = sum(argmax == y) / size(y);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % Y = K x n
    % P = K x n
    % grad_W = K x d
    % grad_b = K x 1
    n = size(x,2);
    G_batch = -(Y - P); % K x n
    grad_W = 1/n * G_batch * X' + 2 * lambda * W; 
    grad_b = 1/n * G_batch * ones(n,1);
end