function Assignment3()
    % Params
    n_batch = 10;
    eta = 0.005;
    n_epochs = 50;
    lambda = 0;
    rng(400);

    input_nodes = 3072;
    output_nodes = 10;
    hidden_nodes = [50,];
    [Ws, bs] = InitializeWeights(input_nodes, output_nodes, hidden_nodes);

    [Wstars, bstars, costs_train, costs_eval, acc_eval] = Train(n_batch, eta, n_epochs, Ws, bs, lambda);

    PlotResults(n_batch, eta, n_epochs, lambda, costs_train, costs_eval, acc_eval);
end

function [Ws, bs] = InitializeWeights(input_nodes, output_nodes, hidden_nodes)
    Ws = {};
    bs = {};

     % Initialize parameters
     %K = size(Y_train, 1); % output_nodes
     %d = size(X_train, 1); % input_nodes
     %W = 0.01 * randn(K, d); % Gaussian random values for W
     %b = 0.01 * randn(K, 1);  % Gaussian random values for b

     if (isempty(hidden_nodes))
        Ws{1} = 0.01 * randn(output_nodes, input_nodes);
        bs{1} = zeros(output_nodes, 1);
        return;
     end

     Ws{1} = 1/sqrt(input_nodes) * randn(hidden_nodes(1), input_nodes);
     bs{1} = zeros(hidden_nodes(1), 1);
     for i=2:length(hidden_nodes)
        Ws{i} = 1/sqrt(hidden_nodes(i-1)) * randn(hidden_nodes(i), hidden_nodes(i-1));
        bs{i} = zeros(hidden_nodes(i),1);
     end
     Ws{length(hidden_nodes)+1} = 1/sqrt(hidden_nodes(end)) * randn(output_nodes, hidden_nodes(end));
     bs{length(hidden_nodes)+1} = zeros(output_nodes, 1);
end

function [Wstars, bstars, costs_train, costs_eval, acc_eval] = Train(n_batch, eta, n_epochs, Ws, bs, lambda)
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

    GDparams = [n_batch, eta, n_epochs];
    [Wstars, bstars, costs_train, costs_eval, acc_eval] = MiniBatchGD(X_train(:,1:100), Y_train(:,1:100), y_train(1:100), X_val, y_val, GDparams, Ws, bs, lambda);

    test_accuracy = ComputeAccuracy(X_test, y_test, Wstars, bstars);
    val_accuracy = ComputeAccuracy(X_val, y_val, Wstars, bstars);
    disp("Test Accuracy: " + test_accuracy);
    disp("Val Accuracy: " + val_accuracy);
end

function PlotResults(n_batch, eta, n_epochs, lambda, costs_train, costs_eval, acc_eval)
    % Plotting
    scr_siz = get(0,'ScreenSize');
    f = figure;
    f.Position = floor([150 150 scr_siz(3)*0.8 scr_siz(4)*0.8]);
    T = tiledlayout(f, 2, 1);
    title(T, "Lambda: " + lambda + ", Epochs: " + n_epochs + ", Batch Size: " + n_batch + ", \eta: " + eta);

    % Plot train-validation losses
    nexttile(T);
    plot(1:n_epochs, costs_train, 1:n_epochs, costs_eval);
    legend("Training loss", "Validation loss");
    %ylim([min(costs_train) * 0.9,max(costs_train) * 1.1]);
    grid();
    xlabel("Epoch")
    ylabel("Loss")
    fontsize(T,24,"points")

    % Plot validation accuracy
    nexttile(T);
    plot(1:n_epochs, acc_eval);
    grid();
    xlabel("Epoch")
    ylabel("Accuracy")
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
    Xs = cell(length(Ws)-1, 1);
    s = X;
    for i=1:length(Ws)-1
        s = Ws{i} * s + bs{i};  % linear transformation
        s(s<0) = 0;             % ReLU activation
        Xs{i} = s;
    end
    s = Ws{end} * s + bs{end};
    P = softmax(s);
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
    regularizationTerm = 0;
    for i=1:length(Ws)
        regularizationTerm = regularizationTerm + lambda * sum(Ws{i}.^2, 'all');
    end

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

function [grad_Ws, grad_bs] = ComputeGradients(Xs, X, Y, P, Ws, bs, lambda)
    grad_Ws = cell(length(Ws),1);
    grad_bs = cell(length(bs),1);
    n = size(X,2);

    G_batch = P-Y;

    for i=length(Ws):-1:2
        grad_Ws{i} = 1/n * G_batch * Xs{i-1}' + 2 * lambda * Ws{i};
        grad_bs{i} = 1/n * G_batch * ones(n,1);
        G_batch = Ws{i}' * G_batch;
        G_batch(Xs{i-1} <= 0) = 0; % not sure if its correct
    end
    grad_Ws{1} = 1/n * G_batch * X' + 2 * lambda * Ws{1};
    grad_bs{1} = 1/n * G_batch * ones(n,1);
end

function [Wstars, bstars, eta] = TrainOneEpoch(X, Y, GDparams, Ws, bs, lambda, epoch) 
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

        [P, Xs] = EvaluateClassifier(Xbatch, Ws, bs);
        [grad_Ws, grad_bs] = ComputeGradients(Xs, Xbatch, Ybatch, P, Ws, bs, lambda);
        for i = 1:length(Ws)
            Ws{i} = Ws{i} - eta * grad_Ws{i};
            bs{i} = bs{i} - eta * grad_bs{i};
        end
    end

    Wstars = Ws;
    bstars = bs;
end

function [Wstars, bstars, costs_train, costs_eval, acc_eval] = MiniBatchGD(X_train, Y_train, y_train, X_val, y_val, GDparams, Ws, bs, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % lambda = scalar
    % GDparams = [n_batch, eta, n_epochs]

    n_epochs = GDparams(3);
    costs_train = zeros(n_epochs, 1);
    costs_eval = zeros(n_epochs, 1);
    acc_eval = zeros(n_epochs, 1);
    for i=1:n_epochs       
        costs_train(i) = ComputeCost(X_train, y_train, Ws, bs, lambda);
        costs_eval(i) = ComputeCost(X_val, y_val, Ws, bs, lambda);
        acc_eval(i) = ComputeAccuracy(X_val, y_val, Ws, bs);
        [Ws, bs, eta] = TrainOneEpoch(X_train, Y_train, GDparams, Ws, bs, lambda, i);
        GDparams(2) = eta;
        disp("Train cost at epoch: " + i + ": " + costs_train(i) + " Eval cost: " + costs_eval(i) + "; eta: " + GDparams(2));
    end 
    
    Wstars = Ws;
    bstars = bs;
end