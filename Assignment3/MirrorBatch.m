function X = MirrorBatch(X_batch, p)
    X = X_batch;
    for i=1:size(X_batch,2)
        if (rand() > p)
            continue;
        end
        aa = 0:1:31;
        bb = 32:-1:1;
        vv = repmat(32*aa, 32, 1);
        ind_flip = vv(:) + repmat(bb', 32, 1);
        inds_flip = [ind_flip; 1024+ind_flip; 2048+ind_flip];

        X(:,i) = X_batch(inds_flip, i);
    end
end