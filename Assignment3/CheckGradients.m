function CheckGradients()
    NetParams = struct();
    NetParams.He_init = true;
    NetParams.use_bn = true; % batch normalization

    % Load data
    [X_train, Y_train, y_train] = LoadBatch("data_batch_1.mat");

    % Preprocess data
    mean_X = mean(X_train, 2);  % d x 1
    std_X = std(X_train, 0, 2); % d x 1

    X_train = NormalizeData(X_train, mean_X, std_X);

    datapoints = 100;
    X_train = X_train(1:10, 1:datapoints);
    Y_train = Y_train(:, 1:datapoints);
    
    n = size(X_train,2);

    NetParams.n = n;
    NetParams.d = 10;
    NetParams.k = size(Y_train, 1);

    % Batch norm stuff
    NetParams.bn_mu = {};
    NetParams.bn_v = {};
    NetParams.bn_alpha = 0.8;

    NetParams.input_nodes = NetParams.d;
    NetParams.output_nodes = NetParams.k;
    NetParams.hidden_nodes = [50, 50, 50];

    NetParams.lambda = 0.2; 

    % Initialize parameters
    [NetParams.W, NetParams.b, NetParams.bn_gamma, NetParams.bn_beta] = InitializeWeights(NetParams);

    n_batch = 10;
    lambda = NetParams.lambda;
    h = 1e-5;

    rel_error_W = zeros(length(NetParams.W), n/n_batch);
    rel_error_b = zeros(length(NetParams.b), n/n_batch);
    rel_error_gamma = zeros(length(NetParams.bn_gamma)-1, n/n_batch);
    rel_error_beta = zeros(length(NetParams.bn_beta)-1, n/n_batch);
    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X_train(:, inds);
        Ybatch = Y_train(:, inds);
        ybatch = y_train(inds);

        [P, Xs, S, S_hat, mu, v, ~] = EvaluateClassifier(Xbatch, NetParams, true);
        [grad_W_a, grad_b_a, grad_gamma_a, grad_beta_a] = ComputeGradients(Xs, S, S_hat, mu, v, Xbatch, Ybatch, P, NetParams);
        grads = ComputeGradsNumSlow(Xbatch, ybatch, NetParams, lambda, h);
        grad_b_n = grads.b;
        grad_W_n = grads.W;

        rel_error_W(:,j) = RelativeError(grad_W_a, grad_W_n, false);
        rel_error_b(:,j) = RelativeError(grad_b_a, grad_b_n, false);

        if NetParams.use_bn
            grad_gamma_n = grads.gammas;
            grad_beta_n = grads.betas;
            rel_error_gamma(:,j) = RelativeError(grad_gamma_a, grad_gamma_n, true);
            rel_error_beta(:,j) = RelativeError(grad_beta_a, grad_beta_n, true);
        end
    end

    figure;
    subplot(2,2,1)
    plot(1:n/n_batch, rel_error_W);
    legend("W layer: 1", "W layer: 2", "W layer: 3", "W layer: 4");
    title("Relative difference for W")

    subplot(2,2,2)
    plot(1:n/n_batch, rel_error_b);
    legend("b layer: 1", "b layer: 2", "b layer: 3", "b layer: 4");
    title("Relative difference for b")

    if NetParams.use_bn
        subplot(2,2,3)
        plot(1:n/n_batch, rel_error_gamma);
        legend("gamma layer: 1", "gamma layer: 2", "gamma layer: 3", "gamma layer: 4");
        title("Relative difference for gamma")

        subplot(2,2,4)
        plot(1:n/n_batch, rel_error_beta);
        legend("beta layer: 1", "beta layer: 2", "beta layer: 3", "beta layer: 4");
        title("Relative difference for beta")
    end
end

function [ret] = RelativeError(X, Y, skiplast)
    n = length(X);
    if skiplast
        n = n - 1;
    end
    ret = zeros(n, 1);
    for j=1:n
        l = zeros(size(X,1));
        eps = 1e-6;
        for i=1:size(X,1) % 1-10
            denom = abs(X{j}(i,:) - Y{j}(i,:));
            denom(denom<eps) = eps;
            l(i) = abs(X{j}(i,:) - Y{j}(i,:)) / denom;
        end
        %ret(j) = mean(l, 'all');
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

function [P, Xs, S, S_hat, mu, v, NetParams] = EvaluateClassifier(X, NetParams, training)
    % X = d x n
    % W = K x d
    % b = K x 1
    % P = K x n

    %s = W * X + b; (K x n)

    % Save
    Xs = cell(length(NetParams.W)-1, 1);
    S = cell(length(NetParams.W)-1, 1);
    S_hat = cell(length(NetParams.W)-1, 1);
    % Save mu and v
    mu = cell(length(NetParams.W)-1,1);
    v = cell(length(NetParams.W)-1,1);

    s = X;

    for i=1:length(NetParams.W)-1
        s = NetParams.W{i} * s + NetParams.b{i};  % linear transformation
        S{i} = s;                                 

        if (training)
            if (NetParams.use_bn)
                mu{i} = mean(s,2);
                v{i} = var(s,0,2) * (size(X,2)-1) / (size(X,2));
                s = diag(v{i} + eps)^(-1/2) * (s-mu{i});
                S_hat{i} = s;
                s = NetParams.bn_gamma{i} .* s + NetParams.bn_beta{i};
            end

            s(s<0) = 0; % ReLU activation
            Xs{i} = s;
        else
            if (NetParams.use_bn && ~isempty(NetParams.bn_mu))
                s = diag(NetParams.bn_v{i} + eps)^(-1/2) * (s-NetParams.bn_mu{i});
                s = NetParams.bn_gamma{i} .* s + NetParams.bn_beta{i};
            end

            s(s<0) = 0; % ReLU activation
            Xs{i} = s;
        end
    end
    s = NetParams.W{end} * s + NetParams.b{end};
    P = softmax(s);

    if (~training)
        return;
    end

    % Save mu and v
    if (NetParams.use_bn && isempty(NetParams.bn_mu))
        NetParams.bn_mu = mu;
        NetParams.bn_v = v;
    elseif (NetParams.use_bn)
        for i=1:length(NetParams.W)-1
            NetParams.bn_mu{i} = NetParams.bn_alpha * NetParams.bn_mu{i} + (1 - NetParams.bn_alpha) * mu{i};
            NetParams.bn_v{i} = NetParams.bn_alpha * NetParams.bn_v{i} + (1 - NetParams.bn_alpha) * v{i};
        end
    end
end

function [grad_Ws, grad_bs, grad_gamma, grad_beta] = ComputeGradients(Xs, S, S_hat, mu, v, X, Y, P, NetParams)
    grad_Ws = cell(length(NetParams.W),1);
    grad_bs = cell(length(NetParams.b),1);
    grad_gamma = cell(length(NetParams.bn_gamma),1);
    grad_beta = cell(length(NetParams.bn_beta),1);
    n = size(X,2);
    k = length(NetParams.W);

    G_batch = P-Y;

    grad_Ws{k} = 1/n * G_batch * Xs{k-1}' + 2 * NetParams.lambda * NetParams.W{k};
    grad_bs{k} = 1/n * G_batch * ones(n,1);

    G_batch = NetParams.W{k}' * G_batch;
    G_batch(Xs{k-1} <= 0) = 0; % not sure if its correct

    for i=length(NetParams.W)-1:-1:1
        if NetParams.use_bn
            grad_gamma{i} = 1/n * (G_batch .* S_hat{i}) * ones(n,1);
            grad_beta{i} = 1/n * G_batch * ones(n,1);
        
            G_batch = G_batch .* (NetParams.bn_gamma{i} * ones(1,n));

            G_batch = BatchNormBackPass(G_batch, S{i}, mu{i}, v{i}, n);
        end

        if (i > 1)
            grad_Ws{i} = 1/n * G_batch * Xs{i-1}' + 2 * NetParams.lambda * NetParams.W{i};
            grad_bs{i} = 1/n * G_batch * ones(n,1);
            G_batch = NetParams.W{i}' * G_batch;
            G_batch(Xs{i-1} <= 0) = 0; % not sure if its correct
        else
            grad_Ws{1} = 1/n * G_batch * X' + 2 * NetParams.lambda * NetParams.W{1};
            grad_bs{1} = 1/n * G_batch * ones(n,1);
        end
    end
end

function [Ws, bs, gamma, beta] = InitializeWeights(NetParams)
    Ws = {};
    bs = {};
    gamma = {};
    beta = {};

    % Initialize parameters
    if (NetParams.He_init)
        sigma = sqrt(2/NetParams.input_nodes);
    else
        sigma = 0.01;
    end

    if (isempty(NetParams.hidden_nodes))
        Ws{1} = sigma * randn(NetParams.output_nodes, NetParams.input_nodes);
        bs{1} = zeros(NetParams.output_nodes, 1);
        return;
    end

    Ws{1} = sigma * randn(NetParams.hidden_nodes(1), NetParams.input_nodes);
    bs{1} = zeros(NetParams.hidden_nodes(1), 1);

    if NetParams.use_bn
        gamma{1} = ones(NetParams.hidden_nodes(1), 1);
        beta{1} = zeros(NetParams.hidden_nodes(1), 1);
    end

    for i=2:length(NetParams.hidden_nodes)
        if (NetParams.He_init)
            sigma = sqrt(2/NetParams.hidden_nodes(i-1));
        else
            sigma = 0.01;
        end
        Ws{i} = sigma * randn(NetParams.hidden_nodes(i), NetParams.hidden_nodes(i-1));
        bs{i} = zeros(NetParams.hidden_nodes(i),1);
        
        if NetParams.use_bn
            gamma{i} = ones(NetParams.hidden_nodes(i),1);
            beta{i} = zeros(NetParams.hidden_nodes(i),1);
        end
    end
    if (NetParams.He_init)
        sigma = sqrt(2/NetParams.hidden_nodes(end));
    else
        sigma = 0.01;
    end
    Ws{length(NetParams.hidden_nodes)+1} = sigma * randn(NetParams.output_nodes, NetParams.hidden_nodes(end));
    bs{length(NetParams.hidden_nodes)+1} = zeros(NetParams.output_nodes, 1);

    if NetParams.use_bn
        gamma{length(NetParams.hidden_nodes)+1} = [];
        beta{length(NetParams.hidden_nodes)+1} = [];
    end
end

function [G_batch] = BatchNormBackPass(G_batch, s, mu, v, n)
    sigma_1 = ((v+eps).^(-0.5));
    sigma_2 = ((v+eps).^(-1.5));
    G_1 = G_batch .* (sigma_1 * ones(1,n));
    G_2 = G_batch .* (sigma_2 * ones(1,n));
    D = s - mu * ones(1,n);
    c = (G_2 .* D) * ones(n,1);
    G_batch = G_1 - 1/n * (G_1 * ones(n,1)) * ones(1,n) - 1/n * D .* (c * ones(1,n));
end