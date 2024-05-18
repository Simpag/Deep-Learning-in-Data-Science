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