function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);

    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));
        
        for i=1:length(b{j})
            
            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);
            
            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);
            
            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));
        
        for i=1:numel(W{j})
            
            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W_try, b, lambda);
        
            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);
        
            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end

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

function J = ComputeCost(X, y, Ws, bs, lambda)
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