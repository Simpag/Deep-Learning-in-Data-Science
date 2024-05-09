function [X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test] = LoadData(NetParams)
    % Load data
    if (NetParams.all_training_data)
        [X1, Y1, y1] = LoadBatch("data_batch_1.mat");
        [X2, Y2, y2] = LoadBatch("data_batch_3.mat");
        [X3, Y3, y3] = LoadBatch("data_batch_4.mat");
        [X4, Y4, y4] = LoadBatch("data_batch_5.mat");
        [X_val, Y_val, y_val] = LoadBatch("data_batch_2.mat");
        [X_test, Y_test, y_test] = LoadBatch("test_batch.mat");
    
        X_train = [X1, X2, X3, X4, X_val(:, 1:end-5000)];
        Y_train = [Y1, Y2, Y3, Y4, Y_val(:, 1:end-5000)];
        y_train = [y1; y2; y3; y4; y_val(1:end-5000)];
    
        X_val = X_val(:, end-4999:end);
        Y_val = Y_val(:, end-4999:end);
        y_val = y_val(end-4999:end);
    else
        [X_train, Y_train, y_train] = LoadBatch("data_batch_1.mat");
        [X_val, Y_val, y_val] = LoadBatch("data_batch_2.mat");
        [X_test, Y_test, y_test] = LoadBatch("test_batch.mat");
    end

    % Preprocess data
    mean_X = mean(X_train, 2);  % d x 1
    std_X = std(X_train, 0, 2); % d x 1

    X_train = NormalizeData(X_train, mean_X, std_X);
    X_val = NormalizeData(X_val, mean_X, std_X);
    X_test = NormalizeData(X_test, mean_X, std_X);
end

function [X, Y, y] = LoadBatch(filename)
    % X contains the image pixel data of size d x n of type double
    % n is the number of images (10'000) and d is the dimensionality of each image (3072 = 32 x 32 x 2)

    % Y is K x n where k is the number of labels (10) and is one-hot encoded of the image label for each image

    % y is a vector of length n containing the label for each image (1-10)

    A = load(filename);
    X = im2double(A.data');
    y = A.labels + 1;
    Y = y == 1:max(y);
    Y = Y';
end

function ret = NormalizeData(X, mean, std)
    ret = X - repmat(mean, [1, size(X, 2)]);
    ret = ret ./ repmat(std, [1, size(ret, 2)]);
end