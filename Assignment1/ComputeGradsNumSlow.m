function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost2(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost2(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end

    for i=1:numel(W)
        
        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost2(X, Y, W_try, b, lambda);
        
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost2(X, Y, W_try, b, lambda);
        
        grad_W(i) = (c2-c1) / (2*h);
    end
end

function P = EvaluateClassifier1(X, W, b)
    % X = d x n
    % W = K x d
    % b = K x 1
    % P = K x n
    s = W * X + b;
    P = softmax(s);
end

function J = ComputeCost1(X, y, W, b, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % Y = K x n
    % J = scalar
    
    P = EvaluateClassifier1(X, W, b);
    n = size(X,2);
    idx = sub2ind(size(P), y', 1:n);
    loss = -1 / n * sum(log(P(idx)));
    regularizationTerm = lambda * sum(W.^2, 'all');

    J = loss + regularizationTerm;
end

% multiple binarcy cross etcx...
function P = EvaluateClassifier2(X, W, b)
    % X = d x n
    % W = K x d
    % b = K x 1
    % P = K x n
    s = W * X + b;
    P = exp(s)./(1+exp(s));
end

function J = ComputeCost2(X, Y, W, b, lambda)
    % X = d x n
    % W = K x d
    % b = K x 1
    % Y = K x n
    % J = scalar
    
    P = EvaluateClassifier2(X, W, b);
    n = size(X,2);
    K = 10;
    %idx = sub2ind(size(P), y', 1:n);
    %loss = -1 / n * sum(log(P(idx)));
    loss = - 1 / n * 1 / K * sum(((1-Y) .* log(1-P)) + (Y .* log(P)), 'all');
    regularizationTerm = lambda * sum(W.^2, 'all');

    J = loss + regularizationTerm;
end