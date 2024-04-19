function CheckGradients()
    % Load data
    [X_train, Y_train, y_train] = LoadBatch("data_batch_1.mat");
    
    % Preprocess data
    mean_X = mean(X_train, 2);  % d x 1
    std_X = std(X_train, 0, 2); % d x 1

    X_train = NormalizeData(X_train, mean_X, std_X);

    % Initialize parameters
    input_nodes = 20;
    output_nodes = 10;
    hidden_nodes = [50,];
    [Ws, bs] = InitializeWeights(input_nodes, output_nodes, hidden_nodes);

    X_train = X_train(1:input_nodes, 1:100);
    Y_train = Y_train(:, 1:100);

    n_batch = 10;
    lambda = 0.0;
    h = 1e-5;
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

        [P, Xs] = EvaluateClassifier(Xbatch, Ws, bs);
        [grad_W_a, grad_b_a] = ComputeGradients(Xs, Xbatch, Ybatch, P, Ws, bs, lambda);
        [grad_b_n, grad_W_n] = ComputeGradsNumSlow(Xbatch, ybatch, Ws, bs, lambda, h);
        for i=1:length(grad_W_a)
            abs_error_W(j) = max(abs_error_W(j), max(abs(grad_W_a{i} - grad_W_n{i}),[],'all'));
            abs_error_b(j) = max(abs_error_b(j), max(abs(grad_b_a{i} - grad_b_n{i}),[],'all'));
        end
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
    for j=1:length(X)
        ret = zeros(size(X,1));
        eps = 1e-6;
        for i=1:size(X,1) % 1-10
            denom = abs(X{j}(i,:) - Y{j}(i,:));
            denom(denom<eps) = eps;
            ret(i) = abs(X{j}(i,:) - Y{j}(i,:)) / denom;
        end
        ret = max(ret, [], 'all');
    end
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