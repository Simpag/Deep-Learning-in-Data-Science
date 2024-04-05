function Assignment1_2()
    % Params
    n_batch = 100;
    eta = 0.01;
    n_epochs = 10;
    lambda = 0.01;
    rng(400);

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

    GDparams = [n_batch, eta, n_epochs];
    [Wstar, bstar, costs_train, costs_eval] = MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, GDparams, W, b, lambda);
    test_accuracy = ComputeAccuracy(X_test, y_test, Wstar, bstar);
    val_accuracy = ComputeAccuracy(X_val, y_val, Wstar, bstar);
    disp("Test Accuracy: " + test_accuracy);
    disp("Val Accuracy: " + val_accuracy)

    scr_siz = get(0,'ScreenSize');
    f = figure;
    f.Position = floor([150 150 scr_siz(3)*0.8 scr_siz(4)*0.8]);
    T = tiledlayout(f, 2,2);
    title(T, "Lambda: " + lambda + ", Epochs: " + n_epochs + ", Batch Size: " + n_batch + ", \eta: " + eta);

    % Visualize the learn weights
    t = tiledlayout(T,4,3);
    s_im = {};
    for i=1:10
        nexttile(t);
        im = reshape(Wstar(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
        imshow(s_im{i})
    end

    nexttile(T);
    plot(1:n_epochs, costs_train, 1:n_epochs, costs_eval);
    legend("Training loss", "Validation loss");
    ylim([min(costs_train) * 0.9,max(costs_train) * 1.1]);
    grid();
    xlabel("Epoch")
    ylabel("Loss")
    fontsize(T,24,"points")

    % Histograms
    nexttile(T, [1,2])
    nbins = 20;
    P = EvaluateClassifier(X_test, Wstar, bstar);
    [~, argmax] = max(P);
    correct = argmax' == y_test;
    hits = [];
    misses = [];
    for i=1:size(correct,1)
        if (correct(i) == 1)
            hits(end+1) = P(argmax(i),i);
        else
            misses(end+1) = P(argmax(i),i);
        end
    end
    histogram(hits,nbins);
    hold on
    histogram(misses,nbins);
    legend("Correctly Classified", "Incorrectly Classified")
    xlim([0,1]);
    xlabel("Probability");
    ylabel("Frequency");
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
    P = exp(s)./(1+exp(s));
end

function J = ComputeCost(X, Y, W, b, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % Y = K x n
    % J = scalar
    
    P = EvaluateClassifier(X, W, b);
    n = size(X,2);
    K = 10;
    %idx = sub2ind(size(P), y', 1:n);
    %loss = -1 / n * sum(log(P(idx)));
    loss = - 1 / n * 1 / K * sum(((1-Y) .* log(1-P)) + (Y .* log(P)), 'all');
    regularizationTerm = lambda * sum(W.^2, 'all');

    J = loss + regularizationTerm;
end

function acc = ComputeAccuracy(X, y, W, b)
    % X = d x n
    % W = K x d
    % b = K x 1
    % y = n x 1
    % acc = scalar

    P = EvaluateClassifier(X, W, b);
    [~, argmax] = max(P);

    acc = sum(argmax' == y) / size(y, 1);
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
    K = 10;
    G_batch = -(Y - P); % K x n
    grad_W = 1/n * 1/K * G_batch * X' + 2 * lambda * W; 
    grad_b = 1/n * 1/K * G_batch * ones(n,1);
end

function [Wstar, bstar, eta] = TrainOneEpoch(X, Y, GDparams, W, b, lambda, epoch) 
    eta = GDparams(2);
    n_batch = GDparams(1);
    n = size(X,2);
    perm = randperm(n);
    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = perm(j_start:j_end);
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);

        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        W = W - eta * grad_W;
        b = b - eta * grad_b;
    end

    Wstar = W;
    bstar = b;
end

function [Wstar, bstar, costs_train, costs_eval] = MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, GDparams, W, b, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % lambda = scalar
    % GDparams = [n_batch, eta, n_epochs]
    n_epochs = GDparams(3);
    costs_train = zeros(n_epochs, 1);
    costs_eval = zeros(n_epochs, 1);
    for i=1:n_epochs       
        costs_train(i) = ComputeCost(X_train, Y_train, W, b, lambda);
        costs_eval(i) = ComputeCost(X_val, Y_val, W, b, lambda);
        [W, b, eta] = TrainOneEpoch(X_train, Y_train, GDparams, W, b, lambda, i);
        disp("Train cost at epoch: " + i + ": " + costs_train(i) + " Eval cost: " + costs_eval(i) + "; eta: " + GDparams(2));
    end 
    
    Wstar = W;
    bstar = b;
end