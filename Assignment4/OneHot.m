function vec = OneHot(i, dims)
    vec = zeros(dims,1);
    vec(i) = 1;
end