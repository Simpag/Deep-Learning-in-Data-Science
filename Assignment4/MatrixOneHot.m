function mat = MatrixOneHot(NetParams, X, char_to_ind)
    mat = zeros(NetParams.K, NetParams.seq_length);

    for i = 1:length(X)
        mat(:,i) = OneHot(char_to_ind(X(i)), NetParams.K);
    end
end