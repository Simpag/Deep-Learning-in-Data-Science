function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)

    Grads.W = cell(numel(NetParams.W), 1);
    Grads.b = cell(numel(NetParams.b), 1);
    if NetParams.use_bn
        Grads.gammas = cell(numel(NetParams.bn_gamma), 1);
        Grads.betas = cell(numel(NetParams.bn_beta), 1);
    end

    for j=1:length(NetParams.b)
        Grads.b{j} = zeros(size(NetParams.b{j}));
        NetTry = NetParams;
        for i=1:length(NetParams.b{j})
            b_try = NetParams.b;
            b_try{j}(i) = b_try{j}(i) - h;
            NetTry.b = b_try;
            c1 = ComputeCost(X, Y, NetTry, lambda);        
            
            b_try = NetParams.b;
            b_try{j}(i) = b_try{j}(i) + h;
            NetTry.b = b_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);
            
            Grads.b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(NetParams.W)
        Grads.W{j} = zeros(size(NetParams.W{j}));
            NetTry = NetParams;
        for i=1:numel(NetParams.W{j})
            
            W_try = NetParams.W;
            W_try{j}(i) = W_try{j}(i) - h;
            NetTry.W = W_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda);
        
            W_try = NetParams.W;
            W_try{j}(i) = W_try{j}(i) + h;
            NetTry.W = W_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);
        
            Grads.W{j}(i) = (c2-c1) / (2*h);
        end
    end

    if NetParams.use_bn
        for j=1:length(NetParams.bn_gamma)
            Grads.gammas{j} = zeros(size(NetParams.bn_gamma{j}));
            NetTry = NetParams;
            for i=1:numel(NetParams.bn_gamma{j})
                
                gammas_try = NetParams.bn_gamma;
                gammas_try{j}(i) = gammas_try{j}(i) - h;
                NetTry.bn_gamma = gammas_try;        
                c1 = ComputeCost(X, Y, NetTry, lambda);
                
                gammas_try = NetParams.bn_gamma;
                gammas_try{j}(i) = gammas_try{j}(i) + h;
                NetTry.bn_gamma = gammas_try;        
                c2 = ComputeCost(X, Y, NetTry, lambda);
                
                Grads.gammas{j}(i) = (c2-c1) / (2*h);
            end
        end
        
        for j=1:length(NetParams.bn_beta)
            Grads.betas{j} = zeros(size(NetParams.bn_beta{j}));
            NetTry = NetParams;
            for i=1:numel(NetParams.bn_beta{j})
                
                betas_try = NetParams.bn_beta;
                betas_try{j}(i) = betas_try{j}(i) - h;
                NetTry.bn_beta = betas_try;        
                c1 = ComputeCost(X, Y, NetTry, lambda);
                
                betas_try = NetParams.bn_beta;
                betas_try{j}(i) = betas_try{j}(i) + h;
                NetTry.bn_beta = betas_try;        
                c2 = ComputeCost(X, Y, NetTry, lambda);
                
                Grads.betas{j}(i) = (c2-c1) / (2*h);
            end
        end    
    end
end

function J = ComputeCost(X, y, NetParams, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % Y = K x n
    % J = scalar
    
    [P, ~, ~, ~, ~, ~, ~] = EvaluateClassifier(X, NetParams, true);
    n = size(X,2);
    idx = sub2ind(size(P), y', 1:n);
    loss = -1 / n * sum(log(P(idx)));
    regularizationTerm = 0;
    for i=1:length(NetParams.W)
        regularizationTerm = regularizationTerm + lambda * sum(NetParams.W{i}.^2, 'all');
    end

    J = loss + regularizationTerm;

    assert(all(size(J) == 1), "Something went wrong when computing the cost function!")
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