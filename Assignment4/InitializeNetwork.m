function RNN = InitializeNetwork(NetParams, sigma)
    RNN = struct();
    RNN.b = zeros(NetParams.m, 1);
    RNN.c = zeros(NetParams.K, 1);
    RNN.U = randn(NetParams.m, NetParams.K) * sigma;
    RNN.W = randn(NetParams.m, NetParams.m) * sigma;
    RNN.V = randn(NetParams.K, NetParams.m) * sigma;
end