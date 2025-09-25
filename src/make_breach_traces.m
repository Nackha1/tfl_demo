function [Br, TR] = make_breach_traces(t, X, Z, sig_names)
% MAKE_BREACH_TRACES  Builds a BreachTraceSystem with signals X and Z.
% sig_names: e.g., {'x','zout'}
if ~exist('BreachTraceSystem','class')
    error('Breach not found on path.');
end
[M, n_traces] = size(X);
assert(all(size(Z)==[M,n_traces]), 'X and Z must be same size.');

Br = BreachTraceSystem(sig_names);
TR = cell(n_traces,1);
for k = 1:n_traces
    tr_mat = [t, X(:,k), Z(:,k)];
    TR{k}  = tr_mat;
    Br.AddTrace(tr_mat);
end
end
