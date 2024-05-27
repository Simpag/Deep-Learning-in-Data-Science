function assignment4_handin()
    rng(1)
    % Load the data
    [data, unique_data, char_to_ind, ind_to_char] = ReadData("data\goblet_book.txt");

    NetParams = struct();

    NetParams.K = length(unique_data);  % Output dimensionality
    NetParams.m = 100;                  % hidden state dimensionality
    NetParams.eta = 0.068129; %0.1      % learning rate
    NetParams.seq_length = 25;          % training sequence length
    NetParams.epochs = 1;

    RNN = InitializeNetwork(NetParams, 0.01);

    [RNN, losses] = Train(RNN, NetParams, data, ind_to_char, char_to_ind);
    PlotResults(losses);
    file_name = strrep("RNN_model_" + string(datetime("now")) + ".xml", " ", "_");
    file_name = strrep(file_name, ":", "-");
    save(file_name, '-struct', 'RNN');

    h_0 = zeros(NetParams.m, 1);
    x_0 = OneHot(char_to_ind("."), NetParams.K);
    disp("Final text generation: ")
    disp(SynthesizeText(RNN, h_0, x_0, 1000, ind_to_char))
end

function grads = Backward(RNN, hs, ps, X, Y)
    grads = struct();

    % Initialize gradients
    grads.U = zeros(size(RNN.U));
    grads.W = zeros(size(RNN.W));
    grads.V = zeros(size(RNN.V));
    grads.b = zeros(size(RNN.b)); 
    grads.c = zeros(size(RNN.c));
    dh_next = zeros(size(hs(:,1)));

    for t = size(X,2):-1:1
        g = ps(:,t) - Y(:,t); % dl / do_t
        grads.V = grads.V + g * hs(:,t+1)'; % dl / dV
        grads.c = grads.c + g;
        dh = RNN.V' * g + dh_next; % dl / dh_t
        da = (1 - hs(:,t+1).^2) .* dh; % dl / da_t
        grads.b = grads.b + da;
        grads.U = grads.U + da * X(:,t)';
        grads.W = grads.W + da * hs(:,t)'; % dl / dW
        dh_next = RNN.W' * da;
    end

    % Clamp gradients
    for f = fieldnames(grads)'
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
end

function [hs, ps, loss] = Forward(RNN, X, Y, h0)
    hs = zeros(size(RNN.W,1), size(X,2)+1);
    ps = zeros(size(RNN.V,1), size(X,2));

    hs(:,1) = h0;
    for t=1:size(X,2)
        a = RNN.W * hs(:,t) + RNN.U * X(:,t) + RNN.b;
        hs(:,t+1) = tanh(a);
        o = RNN.V * hs(:,t+1) + RNN.c;
        ps(:,t) = softmax(o);
    end

    idx = find(Y);
    loss = -sum(log(ps(idx))); %#ok<FNDSB>
end

function RNN = InitializeNetwork(NetParams, sigma)
    RNN = struct();
    RNN.b = zeros(NetParams.m, 1);
    RNN.c = zeros(NetParams.K, 1);
    RNN.U = randn(NetParams.m, NetParams.K) * sigma;
    RNN.W = randn(NetParams.m, NetParams.m) * sigma;
    RNN.V = randn(NetParams.K, NetParams.m) * sigma;
end

function mat = MatrixOneHot(NetParams, X, char_to_ind)
    mat = zeros(NetParams.K, NetParams.seq_length);

    for i = 1:length(X)
        mat(:,i) = OneHot(char_to_ind(X(i)), NetParams.K);
    end
end

function vec = OneHot(i, dims)
    vec = zeros(dims,1);
    vec(i) = 1;
end

function [data, unique_data, char_to_ind, ind_to_char] = ReadData(location)
    fid = fopen(location, 'r');
    data = fscanf(fid,'%c');
    fclose(fid);

    unique_data = unique(data);

    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    ind_to_char = containers.Map('KeyType','int32','ValueType','char');

    for i = 1:length(unique_data)
        char_to_ind(unique_data(i)) = i;
        ind_to_char(i) = unique_data(i);
    end
end

function text = SynthesizeText(RNN, h0, x0, n, ind_to_char)
    text = "";  % Initialize the output text as an empty string
    
    x = x0;    % Start with the initial input
    h = h0;    % Start with the initial hidden state
    for t = 1:n
        % Compute activations
        a = RNN.W * h + RNN.U * x + RNN.b;
        h = tanh(a);
        o = RNN.V * h + RNN.c;
        p = softmax(o);

        % Sampling from the probability distribution
        cp = cumsum(p);
        a = rand();
        ixs = find(cp - a > 0);

        % One-hot encode the sampled index
        x = OneHot(ixs(1), size(x,1));

        % Append the generated character to the text
        text = append(text, ind_to_char(ixs(1)));
    end
end

function [RNN, losses] = Train(RNN, NetParams, data, ind_to_char, char_to_ind)
    losses = zeros(floor(length(data)/NetParams.seq_length) * NetParams.epochs, 1);
    iter = 0;

    adagrad = struct();
    for f = fieldnames(RNN)'
        adagrad.(f{1}) = zeros(size(RNN.(f{1})));
    end

    smooth_loss = -inf;
    for epoch = 1:NetParams.epochs
        h_prev = zeros(NetParams.m, 1);
        e = 1;

        while true
            X_chars = data(e:e+NetParams.seq_length-1);
            X_chars = MatrixOneHot(NetParams, X_chars, char_to_ind);
            Y_chars = data(e+1:e+NetParams.seq_length);
            Y_chars = MatrixOneHot(NetParams, Y_chars, char_to_ind);
        
            [hs, ps, loss] = Forward(RNN, X_chars, Y_chars, h_prev);
            grads = Backward(RNN, hs, ps, X_chars, Y_chars);

            for f = fieldnames(RNN)'
                adagrad.(f{1}) = adagrad.(f{1}) + grads.(f{1}).^2;
                RNN.(f{1}) = RNN.(f{1}) - NetParams.eta ./ sqrt(adagrad.(f{1}) + eps) .* grads.(f{1});
            end

            if smooth_loss == -inf
                smooth_loss = loss;
            else
                smooth_loss = .999 * smooth_loss + 0.001 * loss;
            end

            losses(iter+1) = smooth_loss;

            if mod(iter, 1000) == 0
                disp("Loss " + smooth_loss + "; Step: " + iter + "; Epoch: " + epoch)
            end

            if mod(iter, 10000) == 0
                txt = SynthesizeText(RNN, h_prev, X_chars(:,1), 200, ind_to_char);
                txt = string(iter) + ": " + txt;
                disp(txt)
                writelines(txt, "run_logs.txt", WriteMode="append");
            end

            h_prev = hs(:,end);

            e = e + NetParams.seq_length;
            iter = iter + 1;

            if e > length(data) - NetParams.seq_length - 1
                break;
            end
        end
    end

end

function PlotResults(losses)
    % Plotting
    plot(1:length(losses), losses);
    title("Training Loss");
    ylabel("Loss");
    xlabel("Training Step")
    grid();
end