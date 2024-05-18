function CheckGradients()
    %rng(0)
    [data, unique_data, char_to_ind, ind_to_char] = ReadData("data\goblet_book.txt");

    NetParams = struct();

    NetParams.K = length(unique_data);  % Output dimensionality
    NetParams.m = 5;                    % hidden state dimensionality
    NetParams.eta = 0.1;                % learning rate
    NetParams.seq_length = 25;          % training sequence length

    RNN = InitializeNetwork(NetParams, 0.01);

    X_chars = data(1:NetParams.seq_length);
    X_chars = MatrixOneHot(NetParams, X_chars, char_to_ind);
    Y_chars = data(2:NetParams.seq_length+1);
    Y_chars = MatrixOneHot(NetParams, Y_chars, char_to_ind);
    h0 = zeros(size(RNN.W, 1), 1);

    h = 1e-4;

    [hs, ps, loss] = Forward(RNN, X_chars, Y_chars, h0);
    a_grads = Backward(RNN, hs, ps, X_chars, Y_chars);
    
    num_grads = ComputeGradsNum(X_chars, Y_chars, RNN, h);

    for f = fieldnames(RNN)'
        disp('Computing relative error for')
        disp(['Field name: ' f{1} ]);
        disp(["Max" RelativeError(a_grads.(f{1}), num_grads.(f{1}), true)])
        %disp(["Mean" RelativeError(a_grads.(f{1}), num_grads.(f{1}), false)]);
    end
end

function [err] = RelativeError(X, Y, maximum)
    denom = max(abs(X), abs(Y));
    denom(denom<eps) = eps;
    err = abs(X - Y) ./ denom;
    if maximum
        err = max(err, [], 'all');
    else
        err = mean(err, 'all');
    end
end