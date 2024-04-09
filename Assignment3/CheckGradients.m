function CheckGradients()
    % Load data
    [X_train, Y_train, y_train] = LoadBatch("data_batch_1.mat");
    
    % Preprocess data
    mean_X = mean(X_train, 2);  % d x 1
    std_X = std(X_train, 0, 2); % d x 1

    X_train = NormalizeData(X_train, mean_X, std_X);

    % Initialize parameters
    K = size(Y_train, 1);
    d = size(X_train, 1);
    W = 0.01 * randn(K, d); % Gaussian random values for W
    b = 0.01 * randn(K, 1);  % Gaussian random values for b


    n_batch = 10;
    lambda = 0.1;
    h = 1e-6;
    n = size(X_train,2);
    n = 20;
    abs_error_W = zeros(n/n_batch);
    abs_error_b = zeros(n/n_batch);
    rel_error_W = zeros(n/n_batch);
    rel_error_b = zeros(n/n_batch);
    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X_train(:, inds);
        Ybatch = Y_train(:, inds);
        ybatch = y_train(inds);

        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W_a, grad_b_a] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        [grad_b_n, grad_W_n] = ComputeGradsNumSlow(Xbatch, ybatch, W, b, lambda, h);
        abs_error_W(j) = max(abs(grad_W_a - grad_W_n),[],'all');
        abs_error_b(j) = max(abs(grad_b_a - grad_b_n),[],'all');
        rel_error_W(j) = RelativeError(grad_W_a, grad_W_n);
        rel_error_b(j) = RelativeError(grad_b_a, grad_b_n);
    end

    disp(max(abs_error_W, [], 'all') + " : " + max(abs_error_b, [], 'all') + " : " + max(rel_error_W, [], 'all') + " : " + max(rel_error_b, [], 'all'));
    disp(mean(abs_error_W, 'all') + " : " + mean(abs_error_b, 'all') + " : " + mean(rel_error_W, 'all') + " : " + mean(rel_error_b, 'all'));

    figure;
    subplot(1,2,1)
    plot(1:n/n_batch, abs_error_W, 1:n/n_batch, abs_error_b);
    legend("W abs diff", "b abs diff");
    title("Max Abs Diff")

    subplot(1,2,2)
    plot(1:n/n_batch, rel_error_W, 1:n/n_batch, rel_error_b);
    legend("W rel diff", "b rel diff");
    title("Max Rel Diff")
end

function [ret] = RelativeError(X, Y) 
    ret = zeros(size(X,1));
    eps = 1e-6;
    for i=1:size(X,1) % 1-10
        denom = abs(X(i,:) - Y(i,:));
        denom(denom<eps) = eps;
        ret(i) = abs(X(i,:) - Y(i,:)) / denom;
    end
    ret = max(ret, [], 'all');
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

function ret = NormalizeData(X, mean, std)
    ret = X - repmat(mean, [1, size(X, 2)]);
    ret = ret ./ repmat(std, [1, size(ret, 2)]);
end

function P = EvaluateClassifier(X, W, b)
    % X = d x n
    % W = K x d
    % b = K x 1
    % P = K x n
    s = W * X + b;
    P = softmax(s);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % Y = K x n
    % P = K x n
    % grad_W = K x d
    % grad_b = K x 1
    n = size(X,2);
    G_batch = -(Y - P); % K x n
    grad_W = 1/n * G_batch * X' + 2 * lambda * W; 
    grad_b = 1/n * G_batch * ones(n,1);
end