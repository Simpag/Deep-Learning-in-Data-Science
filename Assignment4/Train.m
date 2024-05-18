function [RNN, losses] = Train(RNN, NetParams, data, ind_to_char, char_to_ind)
    losses = zeros(floor(length(data)/NetParams.seq_length) * NetParams.epochs, 1);
    iter = 0;

    adagrad = struct();
    for f = fieldnames(RNN)'
        adagrad.(f{1}) = zeros(size(RNN.(f{1})));
    end

    for epoch = 1:NetParams.epochs
        smooth_loss = -inf;

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

            if mod(iter, 100) == 0
                disp("Loss " + smooth_loss + "; Step: " + iter + "; Epoch: " + epoch)
            end

            if mod(iter, 500) == 0
                disp(SynthesizeText(RNN, h_prev, X_chars(:,1), 200, ind_to_char))
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