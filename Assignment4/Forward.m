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