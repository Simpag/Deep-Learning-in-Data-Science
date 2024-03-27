A = load("data_batch_1.mat");
I = reshape(A.data', 32, 32, 3, 10000);
I = permute(I, [2, 1, 3, 4]);
montage(I(:,:,:,1:500), 'Size', [5,5]);