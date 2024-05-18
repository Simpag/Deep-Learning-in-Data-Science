function loss = ComputeLoss(X, Y, RNN, h0)    
    [~, ~, loss] = Forward(RNN, X, Y, h0);
end