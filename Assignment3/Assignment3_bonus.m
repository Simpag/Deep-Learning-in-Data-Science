function Assignment3_bonus()
    NetParams = struct();
    NetParams.disable_logging = false;
    NetParams.all_training_data = true;
    NetParams.use_adam = true;
    NetParams.use_gpu = false;
    NetParams.augment_batch = false;
    NetParams.use_dropout = false;
    NetParams.He_init = true;
    NetParams.use_bn = true; % batch normalization

    % Load data
    [X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test] = LoadData(NetParams);
    NetParams.n = size(X_train,2);
    NetParams.d = size(X_train, 1);
    NetParams.k = size(Y_train, 1);

    % Batch norm stuff
    NetParams.bn_mu = {};
    NetParams.bn_v = {};
    NetParams.bn_alpha = 0.9;

    % Params
    NetParams.init_sigma = 1e-4; % only used when not using He_init
    NetParams.shiftProb = 0.1;
    NetParams.mirrorProb = 0.2;

    NetParams.input_nodes = NetParams.d;
    NetParams.output_nodes = NetParams.k;
    NetParams.hidden_nodes = [50, 50, 50, 50, 50, 50];
    NetParams.dropout = 1.0;

    NetParams.n_batch = 100;
    NetParams.n_epochs = 100;
    NetParams.lambda = 0.003; %0.0045418; %optiaml

    NetParams.eta_min = 5e-5; %0.00002;
    NetParams.eta_max = 1e-3; %0.0008;
    NetParams.eta_step = 5 * 45000 / NetParams.n_batch; %2 * floor(NetParams.n / NetParams.n_batch);
    NetParams.eta = 0.002;

    % Adam
    NetParams.beta_1 = 0.9;
    NetParams.beta_2 = 0.999;
    NetParams.epsilon = 1e-8;

    max_num_cycles = 2;
    NetParams.max_train_steps = 2 * NetParams.eta_step * max_num_cycles;
    NetParams.log_frequency = 10; % How many times per cycle to log loss/accuracy etc

    % ParameterSearch(X_train, Y_train, y_train, X_val, y_val, NetParams);
    % return;

    % Init network
    [NetParams.Ws, NetParams.bs, NetParams.bn_gamma, NetParams.bn_beta] = InitializeWeights(NetParams);
    [NetParams, costs_train, costs_val, loss_train, loss_val, acc_train, acc_val, etas, time] = MiniBatchGD(X_train, Y_train, y_train, X_val, y_val, NetParams);
    
    test_accuracy = gather(ComputeAccuracy(X_test, y_test, NetParams));
    val_accuracy = gather(ComputeAccuracy(X_val, y_val, NetParams));
    disp("Test Accuracy: " + test_accuracy);
    disp("Val Accuracy: " + val_accuracy);
    
    PlotResults(NetParams, costs_train, costs_val, loss_train, loss_val, acc_train, acc_val, etas, time, test_accuracy, val_accuracy);
end

function ParameterSearch(X_train, Y_train, y_train, X_val, y_val, NetParams)
    lmin = -5;
    lmax = -1;
    filename = "C:\Users\Simon\Documents\Github\Deep-Learning-in-Data-Science\Assignment3/fine_search.txt";
    grid = logspace(lmin, lmax, 25);

    lmin = log10(0.0014);
    lmax = log10(0.007);

    %for i=grid
    for i=1:25
        tic;
        NP = NetParams;
        l = lmin + (lmax - lmin)*rand(1, 1); 
        NP.lambda = 10^l;
        
        %NP.lambda = i;

        % Init network
        [NP.Ws, NP.bs, NP.bn_gamma, NP.bn_beta] = InitializeWeights(NP);

        [NP, ~, ~, ~, ~, ~, ~, ~, ~] = MiniBatchGD(X_train, Y_train, y_train, X_val, y_val, NP);
        
        val_accuracy = ComputeAccuracy(X_val, y_val, NP);

        data = string(NP.lambda) + ";" + string(val_accuracy) + "\n";

        disp("Tried lambda: " + NP.lambda + "; Accuracy: " + val_accuracy);
        toc;

        writelines(data, filename, WriteMode="append");
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
        sigma = NetParams.init_sigma;
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
            sigma = NetParams.init_sigma;
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
        sigma = NetParams.init_sigma;
    end
    Ws{length(NetParams.hidden_nodes)+1} = sigma * randn(NetParams.output_nodes, NetParams.hidden_nodes(end));
    bs{length(NetParams.hidden_nodes)+1} = zeros(NetParams.output_nodes, 1);

    if NetParams.use_bn
        gamma{length(NetParams.hidden_nodes)+1} = [];
        beta{length(NetParams.hidden_nodes)+1} = [];
    end
end

function [P, Xs, S, S_hat, mu, v, NetParams] = EvaluateClassifier(X, NetParams, training)
    % X = d x n
    % W = K x d
    % b = K x 1
    % P = K x n

    %s = W * X + b; (K x n)

    % Save
    Xs = cell(length(NetParams.Ws)-1, 1);
    S = cell(length(NetParams.Ws)-1, 1);
    S_hat = cell(length(NetParams.Ws)-1, 1);
    % Save mu and v
    mu = cell(length(NetParams.Ws)-1,1);
    v = cell(length(NetParams.Ws)-1,1);

    s = X;
    if (NetParams.use_gpu)
        s = gpuArray(s);
    end
    for i=1:length(NetParams.Ws)-1
        s = NetParams.Ws{i} * s + NetParams.bs{i};  % linear transformation
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
            if (NetParams.use_dropout)
                s = s .* (rand(size(s)) <= NetParams.dropout) / NetParams.dropout;  % Dropout
            end
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
    s = NetParams.Ws{end} * s + NetParams.bs{end};
    P = softmax(s);

    if (~training)
        return;
    end

    % Save mu and v
    if (NetParams.use_bn && isempty(NetParams.bn_mu))
        NetParams.bn_mu = mu;
        NetParams.bn_v = v;
    elseif (NetParams.use_bn)
        for i=1:length(NetParams.Ws)-1
            NetParams.bn_mu{i} = NetParams.bn_alpha * NetParams.bn_mu{i} + (1 - NetParams.bn_alpha) * mu{i};
            NetParams.bn_v{i} = NetParams.bn_alpha * NetParams.bn_v{i} + (1 - NetParams.bn_alpha) * v{i};
        end
    end
end

function [J, loss] = ComputeCost(X, y, NetParams)
    % X = d x n
    % W = K x d
    % b = K x 1
    % Y = K x n
    % J = scalar
    
    [P, ~, ~, ~, ~, ~, ~] = EvaluateClassifier(X, NetParams, false);
    n = size(X,2);
    idx = sub2ind(size(P), y', 1:n);
    loss = -1 / n * sum(log(P(idx)));
    regularizationTerm = 0;
    for i=1:length(NetParams.Ws)
        regularizationTerm = regularizationTerm + NetParams.lambda * sum(NetParams.Ws{i}.^2, 'all');
    end

    J = loss + regularizationTerm;

    assert(all(size(J) == 1), "Something went wrong when computing the cost function!")
end

function acc = ComputeAccuracy(X, y, NetParams)
    % X = d x n
    % W = K x d
    % b = K x 1
    % y = n x 1
    % acc = scalar

    [P, ~, ~, ~, ~, ~, ~] = EvaluateClassifier(X, NetParams, false);
    [~, argmax] = max(P);

    acc = sum(argmax' == y) / size(y, 1);
end

function [grad_Ws, grad_bs, grad_gamma, grad_beta] = ComputeGradients(Xs, S, S_hat, mu, v, X, Y, P, NetParams)
    grad_Ws = cell(length(NetParams.Ws),1);
    grad_bs = cell(length(NetParams.bs),1);
    grad_gamma = cell(length(NetParams.bn_gamma),1);
    grad_beta = cell(length(NetParams.bn_beta),1);
    n = size(X,2);
    k = length(NetParams.Ws);

    G_batch = P-Y;
    if (NetParams.use_gpu)
        G_batch = gpuArray(G_batch);
    end

    grad_Ws{k} = 1/n * G_batch * Xs{k-1}' + 2 * NetParams.lambda * NetParams.Ws{k};
    grad_bs{k} = 1/n * G_batch * ones(n,1);

    G_batch = NetParams.Ws{k}' * G_batch;
    G_batch(Xs{k-1} <= 0) = 0; 

    for i=length(NetParams.Ws)-1:-1:1
        if NetParams.use_bn
            grad_gamma{i} = 1/n * (G_batch .* S_hat{i}) * ones(n,1);
            grad_beta{i} = 1/n * G_batch * ones(n,1);
            
            G_batch = G_batch .* (NetParams.bn_gamma{i} * ones(1,n));

            G_batch = BatchNormBackPass(G_batch, S{i}, mu{i}, v{i}, n);
        end

        if (i > 1)
            grad_Ws{i} = 1/n * G_batch * Xs{i-1}' + 2 * NetParams.lambda * NetParams.Ws{i};
            grad_bs{i} = 1/n * G_batch * ones(n,1);
            G_batch = NetParams.Ws{i}' * G_batch;
            G_batch(Xs{i-1} <= 0) = 0;
        else
            grad_Ws{1} = 1/n * G_batch * X' + 2 * NetParams.lambda * NetParams.Ws{1};
            grad_bs{1} = 1/n * G_batch * ones(n,1);
        end
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

function [NetParams, costs_train, costs_val, loss_train, loss_val, acc_train, acc_val, etas, time] = MiniBatchGD(X_train, Y_train, y_train, X_val, y_val, NetParams)
    % X = d x n
    % W = K x d
    % b = K x 1
    % lambda = scalar

    costs_train = [];
    costs_val = [];
    loss_train = [];
    loss_val = [];
    acc_train = [];
    acc_val = [];
    etas = [];
    time = [];

    adamWs = {};
    adambs = {};
    adamGamma = {};
    adamBeta = {};
    if (NetParams.use_adam)
        for k = 1:length(NetParams.Ws)
            adamWs{k} = AdamOptimizer(NetParams.beta_1, NetParams.beta_2, NetParams.epsilon, size(NetParams.Ws{k},1), size(NetParams.Ws{k},2));
            adambs{k} = AdamOptimizer(NetParams.beta_1, NetParams.beta_2, NetParams.epsilon, size(NetParams.bs{k},1), size(NetParams.bs{k},2));

            if (NetParams.use_bn)
                adamGamma{k} = AdamOptimizer(NetParams.beta_1, NetParams.beta_2, NetParams.epsilon, size(NetParams.bn_gamma{k},1), size(NetParams.bn_gamma{k},2));
                adamBeta{k} = AdamOptimizer(NetParams.beta_1, NetParams.beta_2, NetParams.epsilon, size(NetParams.bn_beta{k},1), size(NetParams.bn_beta{k},2));
            end
        end
    end

    n = size(X_train,2);
    m = n/NetParams.n_batch;
    t = 0;
    idx = 1;
    eta = NetParams.eta;
    for i=1:NetParams.n_epochs       
        perm = randperm(n);
        for j=1:m
            j_start = (j-1)*NetParams.n_batch + 1;
            j_end = j*NetParams.n_batch;
            inds = perm(j_start:j_end);
            Xbatch = X_train(:, inds);
            if NetParams.augment_batch
                Xbatch = AugmentBatch(Xbatch, NetParams);
            end
            Ybatch = Y_train(:, inds);
            
            if (NetParams.use_adam)
                eta = LinearDecay(t, NetParams);
                %eta = StepDecay(t, eta);
            else
                eta = CyclicScheduler(t, NetParams);
            end

            if (mod(t,floor(2 * NetParams.eta_step / NetParams.log_frequency)) == 0)
                if (~NetParams.disable_logging)
                    etas(idx) = eta;
                    [costs_train(idx), loss_train(idx)] = ComputeCost(X_train, y_train, NetParams);
                    [costs_val(idx), loss_val(idx)] = ComputeCost(X_val, y_val, NetParams);
                    acc_train(idx) = ComputeAccuracy(X_train, y_train, NetParams);
                    acc_val(idx) = ComputeAccuracy(X_val, y_val, NetParams);
                    time(idx) = t;

                    disp("Update step: " + time(idx) + "; Train cost: " + costs_train(idx) + "; Eval cost: " + costs_val(idx) + "; Val acc: " + acc_val(idx) + "; Train acc: " + acc_train(idx));
                    idx = idx + 1;
                else
                    disp("Update step: " + t)
                end                
            end

            t = t + 1;

            [P, Xs, S, S_hat, mu, v, NetParams] = EvaluateClassifier(Xbatch, NetParams, true);
            [grad_Ws, grad_bs, grad_gamma, grad_beta] = ComputeGradients(Xs, S, S_hat, mu, v, Xbatch, Ybatch, P, NetParams);
            if (NetParams.use_adam)
                for k = 1:length(NetParams.Ws)
                    [adamWs{k}, NetParams.Ws{k}] = adamWs{k}.Update(NetParams.Ws{k}, grad_Ws{k}, eta);
                    [adambs{k}, NetParams.bs{k}] = adambs{k}.Update(NetParams.bs{k}, grad_bs{k}, eta);

                    if (NetParams.use_bn)
                        [adamGamma{k}, NetParams.bn_gamma{k}] = adamGamma{k}.Update(NetParams.bn_gamma{k}, grad_gamma{k}, eta);
                        [adamBeta{k}, NetParams.bn_beta{k}] = adamBeta{k}.Update(NetParams.bn_beta{k}, grad_beta{k}, eta);
                    end
                end
            else
                for k = 1:length(NetParams.Ws)
                    NetParams.Ws{k} = NetParams.Ws{k} - eta * grad_Ws{k};
                    NetParams.bs{k} = NetParams.bs{k} - eta * grad_bs{k};

                    if (NetParams.use_bn)
                        NetParams.bn_gamma{k} = NetParams.bn_gamma{k} - eta * grad_gamma{k};
                        NetParams.bn_beta{k} = NetParams.bn_beta{k} - eta * grad_beta{k};
                    end
                end
            end
            
            if (t >= NetParams.max_train_steps && NetParams.max_train_steps > 0) % exit training...
                if (~NetParams.disable_logging)
                    etas(idx) = eta;
                    [costs_train(idx), loss_train(idx)] = ComputeCost(X_train, y_train, NetParams);
                    [costs_val(idx), loss_val(idx)] = ComputeCost(X_val, y_val, NetParams);
                    acc_train(idx) = ComputeAccuracy(X_train, y_train, NetParams);
                    acc_val(idx) = ComputeAccuracy(X_val, y_val, NetParams);
                    time(idx) = t;

                    disp("Update step: " + time(idx) + "; Train cost: " + costs_train(idx) + "; Eval cost: " + costs_val(idx) + "; Val acc: " + acc_val(idx) + "; Train acc: " + acc_train(idx));
                else
                    disp("Update step: " + t)
                end

                return;
            end
        end
    end 
end

function X = AugmentBatch(X_batch, NetParams)
    X = MirrorBatch(X_batch, NetParams.mirrorProb);
    X = ShiftBatch(X, NetParams.shiftProb);
end

function eta = CyclicScheduler(t, NetParams)
    t = mod(t, 2*NetParams.eta_step);
    if ((0 <= t) && (t <= NetParams.eta_step))
        eta = NetParams.eta_min + t / NetParams.eta_step * (NetParams.eta_max - NetParams.eta_min);
    else
        eta = NetParams.eta_max - (t-NetParams.eta_step) / NetParams.eta_step * (NetParams.eta_max - NetParams.eta_min);
    end
end

function eta = LinearDecay(t, NetParams)
    eta = NetParams.eta_max - (NetParams.eta_max - NetParams.eta_min) * t / NetParams.max_train_steps;
end

function eta = StepDecay(t, eta)
    if (mod(t,2500) == 0)
        eta = eta * 0.95;
    end
end