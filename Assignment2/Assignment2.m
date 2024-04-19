function Assignment2()
    NetParams = struct();
    NetParams.disable_logging = false;

    % Load data
    [X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test] = LoadData();
    NetParams.n = size(X_train,2);
    NetParams.d = size(X_train, 1);
    NetParams.k = size(Y_train, 1);

    % Params
    NetParams.input_nodes = NetParams.d;
    NetParams.output_nodes = NetParams.k;
    NetParams.hidden_nodes = [50,];

    NetParams.n_batch = 100;
    NetParams.n_epochs = 100;
    NetParams.lambda = 0.0028884; % best lambda

    NetParams.eta_min = 1e-5;
    NetParams.eta_max = 1e-1;
    NetParams.eta_step = 2 * floor(NetParams.n / NetParams.n_batch);
    NetParams.eta = 0.005;

    max_num_cycles = 5;
    NetParams.max_train_steps = 2 * NetParams.eta_step * max_num_cycles; % Last multi is number of cycles
    NetParams.log_frequency = 10; % How many times per cycle to log loss/accuracy etc
    %rng(400);

    %ParameterSearch(X_train, Y_train, y_train, X_val, y_val, NetParams);
    %return;

    % Init network
    [NetParams.Ws, NetParams.bs] = InitializeWeights(NetParams);

    [NetParams.Ws, NetParams.bs, costs_train, costs_val, loss_train, loss_val, acc_train, acc_val, etas, time] = MiniBatchGD(X_train, Y_train, y_train, X_val, y_val, NetParams);
    
    test_accuracy = ComputeAccuracy(X_test, y_test, NetParams.Ws, NetParams.bs);
    val_accuracy = ComputeAccuracy(X_val, y_val, NetParams.Ws, NetParams.bs);
    disp("Test Accuracy: " + test_accuracy);
    disp("Val Accuracy: " + val_accuracy);

    PlotResults(NetParams, costs_train, costs_val, loss_train, loss_val, acc_train, acc_val, etas, time);
end

function ParameterSearch(X_train, Y_train, y_train, X_val, y_val, NetParams)
    lmin = -5;
    lmax = -1;
    filename = "/Users/mikaelgranstrom/Documents/github/Deep-Learning-in-Data-Science/Assignment2/fine_search.txt";
    grid = logspace(lmin, lmax, 10);

    lmin = log10(0.0006);
    lmax = log10(0.01);

    %for i=grid
    for i=1:20
        NP = NetParams;
        l = lmin + (lmax - lmin)*rand(1, 1); 
        NP.lambda = 10^l;
        
        %NP.lambda = i;

        % Init network
        [NP.Ws, NP.bs] = InitializeWeights(NP);

        [NP.Ws, NP.bs, ~, ~, ~, ~, ~, ~, ~, ~] = MiniBatchGD(X_train, Y_train, y_train, X_val, y_val, NP);
        
        val_accuracy = ComputeAccuracy(X_val, y_val, NP.Ws, NP.bs);

        data = string(NP.lambda) + ";" + string(val_accuracy) + "\n";

        disp("Tried lambda: " + NP.lambda + "; Accuracy: " + val_accuracy);

        writelines(data, filename, WriteMode="append");
    end
end

function [Ws, bs] = InitializeWeights(NetParams)
    Ws = {};
    bs = {};

     % Initialize parameters
     %K = size(Y_train, 1); % output_nodes
     %d = size(X_train, 1); % input_nodes
     %W = 0.01 * randn(K, d); % Gaussian random values for W
     %b = 0.01 * randn(K, 1);  % Gaussian random values for b

     if (isempty(NetParams.hidden_nodes))
        Ws{1} = 0.01 * randn(NetParams.output_nodes, NetParams.input_nodes);
        bs{1} = zeros(NetParams.output_nodes, 1);
        return;
     end

     Ws{1} = 1/sqrt(NetParams.input_nodes) * randn(NetParams.hidden_nodes(1), NetParams.input_nodes);
     bs{1} = zeros(NetParams.hidden_nodes(1), 1);
     for i=2:length(NetParams.hidden_nodes)
        Ws{i} = 1/sqrt(NetParams.hidden_nodes(i-1)) * randn(NetParams.hidden_nodes(i), NetParams.hidden_nodes(i-1));
        bs{i} = zeros(NetParams.hidden_nodes(i),1);
     end
     Ws{length(NetParams.hidden_nodes)+1} = 1/sqrt(NetParams.hidden_nodes(end)) * randn(NetParams.output_nodes, NetParams.hidden_nodes(end));
     bs{length(NetParams.hidden_nodes)+1} = zeros(NetParams.output_nodes, 1);
end

function [X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test] = LoadData()
    % Load data
    [X1, Y1, y1] = LoadBatch("data_batch_1.mat");
    [X2, Y2, y2] = LoadBatch("data_batch_3.mat");
    [X3, Y3, y3] = LoadBatch("data_batch_4.mat");
    [X4, Y4, y4] = LoadBatch("data_batch_5.mat");
    %[X_train, Y_train, y_train] = LoadBatch("data_batch_1.mat");
    [X_val, Y_val, y_val] = LoadBatch("data_batch_2.mat");
    [X_test, Y_test, y_test] = LoadBatch("test_batch.mat");

    X_train = [X1, X2, X3, X4, X_val(:, 1:end-5000)];
    Y_train = [Y1, Y2, Y3, Y4, Y_val(:, 1:end-5000)];
    y_train = [y1; y2; y3; y4; y_val(1:end-5000)];

    X_val = X_val(:, end-4999:end);
    Y_val = Y_val(:, end-4999:end);
    y_val = y_val(end-4999:end);

    % Preprocess data
    mean_X = mean(X_train, 2);  % d x 1
    std_X = std(X_train, 0, 2); % d x 1

    X_train = NormalizeData(X_train, mean_X, std_X);
    X_val = NormalizeData(X_val, mean_X, std_X);
    X_test = NormalizeData(X_test, mean_X, std_X);
end

function PlotResults(NetParams, costs_train, costs_val, loss_train, loss_val, acc_train, acc_val, etas, time)
    % Plotting
    scr_siz = get(0,'ScreenSize');
    f = figure;
    f.Position = floor([150 150 scr_siz(3)*0.8 scr_siz(4)*0.8]);
    T = tiledlayout(f, 2, 2);
    title(T, "Lambda: " + NetParams.lambda + ", Epochs: " + NetParams.n_epochs + ", Batch Size: " + NetParams.n_batch);

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

function [J, loss] = ComputeCost(X, y, Ws, bs, lambda)
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

function [Wstars, bstars, costs_train, costs_val, loss_train, loss_val, acc_train, acc_val, etas, time] = MiniBatchGD(X_train, Y_train, y_train, X_val, y_val, NetParams)
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
            Ybatch = Y_train(:, inds);
            eta = CyclicScheduler(t, NetParams);

            if (mod(t,floor(2 * NetParams.eta_step / NetParams.log_frequency)) == 0)
                if (~NetParams.disable_logging)
                    etas(idx) = eta;
                    [costs_train(idx), loss_train(idx)] = ComputeCost(X_train, y_train, NetParams.Ws, NetParams.bs, NetParams.lambda);
                    [costs_val(idx), loss_val(idx)] = ComputeCost(X_val, y_val, NetParams.Ws, NetParams.bs, NetParams.lambda);
                    acc_train(idx) = ComputeAccuracy(X_train, y_train, NetParams.Ws, NetParams.bs);
                    acc_val(idx) = ComputeAccuracy(X_val, y_val, NetParams.Ws, NetParams.bs);
                    time(idx) = t;

                    disp("Train cost at update step: " + time(idx) + ": " + costs_train(idx) + " Eval cost: " + costs_val(idx));
                    idx = idx + 1;
                else
                    disp("Update step: " + t)
                end                
            end

            t = t + 1;

            [P, Xs] = EvaluateClassifier(Xbatch, NetParams.Ws, NetParams.bs);
            [grad_Ws, grad_bs] = ComputeGradients(Xs, Xbatch, Ybatch, P, NetParams.Ws, NetParams.bs, NetParams.lambda);
            for k = 1:length(NetParams.Ws)
                NetParams.Ws{k} = NetParams.Ws{k} - eta * grad_Ws{k};
                NetParams.bs{k} = NetParams.bs{k} - eta * grad_bs{k};
            end

            if (t >= NetParams.max_train_steps && NetParams.max_train_steps > 0 && ~NetParams.disable_logging)
                etas(idx) = eta;
                [costs_train(idx), loss_train(idx)] = ComputeCost(X_train, y_train, NetParams.Ws, NetParams.bs, NetParams.lambda);
                [costs_val(idx), loss_val(idx)] = ComputeCost(X_val, y_val, NetParams.Ws, NetParams.bs, NetParams.lambda);
                acc_train(idx) = ComputeAccuracy(X_train, y_train, NetParams.Ws, NetParams.bs);
                acc_val(idx) = ComputeAccuracy(X_val, y_val, NetParams.Ws, NetParams.bs);
                time(idx) = t;

                Wstars = NetParams.Ws;
                bstars = NetParams.bs;

                return;
            end
        end
    end 

    Wstars = NetParams.Ws;
    bstars = NetParams.bs;
end

function eta = CyclicScheduler(t, NetParams)
    t = mod(t, 2*NetParams.eta_step);
    if ((0 <= t) && (t <= NetParams.eta_step))
        eta = NetParams.eta_min + t / NetParams.eta_step * (NetParams.eta_max - NetParams.eta_min);
    else
        eta = NetParams.eta_max - (t-NetParams.eta_step) / NetParams.eta_step * (NetParams.eta_max - NetParams.eta_min);
    end
end