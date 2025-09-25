function [idx_train, idx_test] = split_train_test(labels, train_ratio)
% SPLIT_TRAIN_TEST  Stratified split (by labels).
n = numel(labels);
idx_all  = (1:n).';
idx_norm = idx_all(labels==0);
idx_ring = idx_all(labels==1);

idx_norm = idx_norm(randperm(numel(idx_norm)));
idx_ring = idx_ring(randperm(numel(idx_ring)));

ntr_norm = max(1, round(train_ratio*numel(idx_norm)));
ntr_ring = max(0, round(train_ratio*numel(idx_ring)));

idx_train = [idx_norm(1:ntr_norm); idx_ring(1:ntr_ring)];
idx_test  = setdiff(idx_all, idx_train);
end
