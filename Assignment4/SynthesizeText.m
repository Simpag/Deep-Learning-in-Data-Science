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