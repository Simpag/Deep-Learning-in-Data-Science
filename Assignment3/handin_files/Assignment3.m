function Assignment3()
    NetParams = struct();
    NetParams.disable_logging = false;
    NetParams.all_training_data = true;
    NetParams.use_adam = false;
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
    NetParams.hidden_nodes = [50, 50];
    NetParams.dropout = 1.0;

    NetParams.n_batch = 100;
    NetParams.n_epochs = 100;
    NetParams.lambda = 0.005; %0.0045418; %optiaml

    NetParams.eta_min = 1e-5;
    NetParams.eta_max = 1e-1;
    NetParams.eta_step = 5 * 45000 / NetParams.n_batch; %2 * floor(NetParams.n / NetParams.n_batch);
    NetParams.eta = 0.0002;

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
                adamBeta{k} = AdamOptimizer(NetParams.beta_1, NetParams.beta_2, NetParams.epsilon, size(NetParams.bn_alpha{k},1), size(NetParams.bn_alpha{k},2));
            end
        end
    end

    n = size(X_train,2);
    m = n/NetParams.n_batch;
    t = 0;
    idx = 1;
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
                eta = NetParams.eta;
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

function X = ShiftBatch(X_batch, p)
    X = X_batch;
    for i=1:size(X_batch,2)
        if (rand() > p)
            continue;
        end
        tx = round(rand() * 10);
        ty = round(rand() * 10);
        aa = 0:1:31;
        vv = repmat(32*aa, 32-tx, 1);
        bb1 = tx+1:1:32;
        bb2 = 1:32-tx;

        ind_fill = vv(:) + repmat(bb1', 32, 1);
        ind_xx = vv(:) + repmat(bb2', 32, 1);

        ii = find(ind_fill >= ty*32+1);
        ind_fill = ind_fill(ii(1):end);

        ii = find(ind_xx <= 1024-ty*32);
        ind_xx = ind_xx(1:ii(end));
        inds_fill = [ind_fill; 1024+ind_fill; 2048+ind_fill];
        inds_xx = [ind_xx; 1024+ind_xx; 2048+ind_xx];

        X(inds_fill,i) = X_batch(inds_xx,i);
    end
end

function X = MirrorBatch(X_batch, p)
    X = X_batch;
    for i=1:size(X_batch,2)
        if (rand() > p)
            continue;
        end
        aa = 0:1:31;
        bb = 32:-1:1;
        vv = repmat(32*aa, 32, 1);
        ind_flip = vv(:) + repmat(bb', 32, 1);
        inds_flip = [ind_flip; 1024+ind_flip; 2048+ind_flip];

        X(:,i) = X_batch(inds_flip, i);
    end
end

function [X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test] = LoadData(NetParams)
    % Load data
    if (NetParams.all_training_data)
        [X1, Y1, y1] = LoadBatch("data_batch_1.mat");
        [X2, Y2, y2] = LoadBatch("data_batch_3.mat");
        [X3, Y3, y3] = LoadBatch("data_batch_4.mat");
        [X4, Y4, y4] = LoadBatch("data_batch_5.mat");
        [X_val, Y_val, y_val] = LoadBatch("data_batch_2.mat");
        [X_test, Y_test, y_test] = LoadBatch("test_batch.mat");
    
        X_train = [X1, X2, X3, X4, X_val(:, 1:end-5000)];
        Y_train = [Y1, Y2, Y3, Y4, Y_val(:, 1:end-5000)];
        y_train = [y1; y2; y3; y4; y_val(1:end-5000)];
    
        X_val = X_val(:, end-4999:end);
        Y_val = Y_val(:, end-4999:end);
        y_val = y_val(end-4999:end);
    else
        [X_train, Y_train, y_train] = LoadBatch("data_batch_1.mat");
        [X_val, Y_val, y_val] = LoadBatch("data_batch_2.mat");
        [X_test, Y_test, y_test] = LoadBatch("test_batch.mat");
    end

    % Preprocess data
    mean_X = mean(X_train, 2);  % d x 1
    std_X = std(X_train, 0, 2); % d x 1

    X_train = NormalizeData(X_train, mean_X, std_X);
    X_val = NormalizeData(X_val, mean_X, std_X);
    X_test = NormalizeData(X_test, mean_X, std_X);
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

function PlotResults(NetParams, costs_train, costs_val, loss_train, loss_val, acc_train, acc_val, etas, time, test_accuracy, val_accuracy)
    if (length(costs_train) < 1)
        disp("Logging must be on to plot!")
        return;
    end

    % Plotting
    scr_siz = get(0,'ScreenSize');
    f = figure;
    f.Position = floor([150 150 scr_siz(3)*0.8 scr_siz(4)*0.8]);
    T = tiledlayout(f, 2, 2);
    title(T, "Test accuracy: " + test_accuracy + ", Validation accuracy: " + val_accuracy);

    % Plot train-validation losses
    nexttile(T);
    plot(time, loss_train, time, loss_val);
    legend("Training loss", "Validation loss");
    %ylim([min(loss_train) * 0.9,max(loss_train) * 1.1]);
    grid();
    xlabel("Update step")
    ylabel("Loss")
    fontsize(T,24,"points")

    % Plot train-validation costs
    nexttile(T);
    plot(time, costs_train, time, costs_val);
    legend("Training cost", "Validation cost");
    %ylim([min(costs_train) * 0.9,max(costs_train) * 1.1]);
    grid();
    xlabel("Update step")
    ylabel("Cost")
    fontsize(T,24,"points")

    % Plot train-validation accuracy
    nexttile(T);
    plot(time, acc_train, time, acc_val);
    legend("Training accuracy", "Validation accuracy");
    grid();
    xlabel("Update step")
    ylabel("Accuracy")
    fontsize(T,24,"points")

    % Plot learning rate
    nexttile(T);
    plot(time, etas);
    grid();
    xlabel("Update step")
    ylabel("\eta")
    fontsize(T,24,"points")
end