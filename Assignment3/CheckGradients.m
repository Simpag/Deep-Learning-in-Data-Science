function CheckGradients()
    NetParams = struct();

    % Load data
    [X_train, Y_train, y_train] = LoadBatch("data_batch_1.mat");
    
    % Preprocess data
    mean_X = mean(X_train, 2);  % d x 1
    std_X = std(X_train, 0, 2); % d x 1

    X_train = NormalizeData(X_train, mean_X, std_X);

    % Initialize parameters
    datapoints = 500;
    input_nodes = 10;
    output_nodes = 10;
    hidden_nodes = [50,50,50];
    [NetParams.W, NetParams.b] = InitializeWeights(input_nodes, output_nodes, hidden_nodes);
    NetParams.use_bn = false;

    X_train = X_train(1:input_nodes, 1:datapoints);
    Y_train = Y_train(:, 1:datapoints);

    n_batch = 20;
    lambda = 0.0;
    h = 1e-5;
    n = size(X_train,2);
    %n = 20;

    rel_error_W = zeros(length(NetParams.W), n/n_batch);
    rel_error_b = zeros(length(NetParams.b), n/n_batch);
    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X_train(:, inds);
        Ybatch = Y_train(:, inds);
        ybatch = y_train(inds);

        [P, Xs] = EvaluateClassifier(Xbatch, NetParams.W, NetParams.b);
        [grad_W_a, grad_b_a] = ComputeGradients(Xs, Xbatch, Ybatch, P, NetParams.W, NetParams.b, lambda);
        grads = ComputeGradsNumSlow(Xbatch, ybatch, NetParams, lambda, h);
        grad_b_n = grads.b;
        grad_W_n = grads.W;

        rel_error_W(:,j) = RelativeError(grad_W_a, grad_W_n);
        rel_error_b(:,j) = RelativeError(grad_b_a, grad_b_n);
    end

    figure;
    subplot(1,2,1)
    plot(1:n/n_batch, rel_error_W);
    legend("W layer: 1", "W layer: 2", "W layer: 3", "W layer: 4");
    title("Relative difference for W")

    subplot(1,2,2)
    plot(1:n/n_batch, rel_error_b);
    legend("b layer: 1", "b layer: 2", "b layer: 3", "b layer: 4");
    title("Relative difference for b")

end

function [ret] = RelativeError(X, Y)
    ret = zeros(length(X), 1);

    for j=1:length(X)
        l = zeros(size(X,1));
        eps = 1e-6;
        for i=1:size(X,1) % 1-10
            denom = abs(X{j}(i,:) - Y{j}(i,:));
            denom(denom<eps) = eps;
            l(i) = abs(X{j}(i,:) - Y{j}(i,:)) / denom;
        end
        ret(j) = max(l, [], 'all');
    end
end

function [X, Y, y] = LoadBatch(filename)
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
        s = Ws{i} * s + bs{i};                        % linear transformation
        s(s<0) = 0;                                                       % ReLU activation
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