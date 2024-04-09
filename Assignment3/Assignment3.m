function Assignment3()
    % Params
    n_batch = 100;
    eta = 0.001;
    n_epochs = 30;
    lambda = 0.01;
    rng(400);

    [Wstar, bstar, costs_train, costs_eval] = Train(n_batch, eta, n_epochs, lambda);
    test_accuracy = ComputeAccuracy(X_test, y_test, Wstar, bstar);
    val_accuracy = ComputeAccuracy(X_val, y_val, Wstar, bstar);
    disp("Test Accuracy: " + test_accuracy);
    disp("Val Accuracy: " + val_accuracy);

    PlotResults(n_batch, eta, n_epochs, lambda, Wstar, costs_train, costs_eval);
end

function [Wstar, bstar, costs_train, costs_eval] = Train(n_batch, eta, n_epochs, lambda)
    % Load data
    %[X1, Y1, y1] = LoadBatch("data_batch_1.mat");
    %[X2, Y2, y2] = LoadBatch("data_batch_3.mat");
    %[X3, Y3, y3] = LoadBatch("data_batch_4.mat");
    %[X4, Y4, y4] = LoadBatch("data_batch_5.mat");
    [X_train, Y_train, y_train] = LoadBatch("data_batch_1.mat");
    [X_val, Y_val, y_val] = LoadBatch("data_batch_2.mat");
    [X_test, Y_test, y_test] = LoadBatch("test_batch.mat");

    %X_train = [X1, X2, X3, X4, X_val(:, 1:end-1000)];
    %Y_train = [Y1, Y2, Y3, Y4, Y_val(:, 1:end-1000)];
    %y_train = [y1; y2; y3; y4; y_val(1:end-1000)];

    %X_val = X_val(:, 1:1000);
    %Y_val = Y_val(:, 1:1000);
    %y_val = y_val(1:1000);

    % Preprocess data
    mean_X = mean(X_train, 2);  % d x 1
    std_X = std(X_train, 0, 2); % d x 1

    X_train = NormalizeData(X_train, mean_X, std_X);
    X_val = NormalizeData(X_val, mean_X, std_X);
    X_test = NormalizeData(X_test, mean_X, std_X);

    %[X_train, Y_train, y_train] = AddFlips(X_train, Y_train, y_train);
    %X_train = AddRandomNoise(X_train, 0.005);

    % Initialize parameters
    K = size(Y_train, 1);
    d = size(X_train, 1);
    W = 0.01 * randn(K, d); % Gaussian random values for W
    b = 0.01 * randn(K, 1);  % Gaussian random values for b

    GDparams = [n_batch, eta, n_epochs];
    [Wstar, bstar, costs_train, costs_eval] = MiniBatchGD(X_train, Y_train, y_train, X_val, y_val, GDparams, W, b, lambda);
end

function PlotResults(n_batch, eta, n_epochs, lambda, Wstar, costs_train, costs_eval)
    scr_siz = get(0,'ScreenSize');
    f = figure;
    f.Position = floor([150 150 scr_siz(3)*0.8 scr_siz(4)*0.8]);
    T = tiledlayout(f, 1,2);
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

function [P, Xs] = EvaluateClassifier(X, Ws, bs)
    % X = d x n
    % W = K x d
    % b = K x 1
    % P = K x n

    %s = W * X + b;

    % Save Xs!
    s = X;
    for i=1:size(Ws)-1
        s = Ws(i) * s + bs(i);  % linear transformation
        s(s<0) = 0;             % ReLU activation
    end
    s = Ws(end) * s + bs(end);
    P = softmax(s);
    assert(all(size(P) == size(X)), "Something went wrong when evaluating the classifier!"); 
end

function J = ComputeCost(X, y, Ws, bs, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % Y = K x n
    % J = scalar
    
    [P, ~] = EvaluateClassifier(X, Ws, bs);
    n = size(X,2);
    idx = sub2ind(size(P), y', 1:n);
    loss = -1 / n * sum(log(P(idx)));
    regularizationTerm = lambda * sum(W.^2, 'all');

    J = loss + regularizationTerm;

    assert(all(size(J) == 1), "Something went wrong when computing the cost function!")
end

function acc = ComputeAccuracy(X, y, Ws, bs)
    % X = d x n
    % W = K x d
    % b = K x 1
    % y = n x 1
    % acc = scalar

    [P, ~] = EvaluateClassifier(X, Ws, bs);
    [~, argmax] = max(P);

    acc = sum(argmax' == y) / size(y, 1);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, Ws, lambda)
    

    %n = size(X,2);
    %G_batch = -(Y - P); % K x n
    %grad_W = 1/n * G_batch * X' + 2 * lambda * W; 
    %grad_b = 1/n * G_batch * ones(n,1);
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

        eta = RampUpStepScheduler(eta, j+(epoch-1)*n/n_batch);
    end

    Wstar = W;
    bstar = b;
end

function [Wstar, bstar, costs_train, costs_eval] = MiniBatchGD(X_train, Y_train, y_train, X_val, y_val, GDparams, W, b, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % lambda = scalar
    % GDparams = [n_batch, eta, n_epochs]
    n_epochs = GDparams(3);
    costs_train = zeros(n_epochs, 1);
    costs_eval = zeros(n_epochs, 1);
    for i=1:n_epochs       
        costs_train(i) = ComputeCost(X_train, y_train, W, b, lambda);
        costs_eval(i) = ComputeCost(X_val, y_val, W, b, lambda);
        [W, b, eta] = TrainOneEpoch(X_train, Y_train, GDparams, W, b, lambda, i);
        GDparams(2) = eta;
        disp("Train cost at epoch: " + i + ": " + costs_train(i) + " Eval cost: " + costs_eval(i) + "; eta: " + GDparams(2));
    end 
    
    Wstar = W;
    bstar = b;
end

function [new_eta] = StepScheduler(eta, step)
    new_eta = eta;
    if (mod(step, 8000) == 0)
        new_eta = eta * 0.8;
    end
end

function [new_eta] = RampUpStepScheduler(eta, step)
    new_eta = eta;

    if (eta <= 0.0001)
        new_eta = 0.0001;
        return;
    end

    if (step < 3000)
        new_eta = eta + 3*1e-6;
        return;
    end

    if (mod(step, 3000) == 0)
        new_eta = eta * 0.5;
    end
end