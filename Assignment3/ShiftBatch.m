function X = ShiftBatch(X_batch, p)
    X = X_batch;
    for i=1:size(X_batch,2)
        if (rand() > p)
            continue;
        end
        tx = round(rand() * 10);
        ty = round(rand() * 10);
        aa = 0:1:31;
        vv = repmat(32*aa, 32-tx, 1);
        bb1 = tx+1:1:32;
        bb2 = 1:32-tx;

        ind_fill = vv(:) + repmat(bb1', 32, 1);
        ind_xx = vv(:) + repmat(bb2', 32, 1);

        ii = find(ind_fill >= ty*32+1);
        ind_fill = ind_fill(ii(1):end);

        ii = find(ind_xx <= 1024-ty*32);
        ind_xx = ind_xx(1:ii(end));
        inds_fill = [ind_fill; 1024+ind_fill; 2048+ind_fill];
        inds_xx = [ind_xx; 1024+ind_xx; 2048+ind_xx];

        X(inds_fill,i) = X_batch(inds_xx,i);
    end
end