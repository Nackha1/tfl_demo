function Br_sub = br_subset_from_idx(TR, sig_names, idx)
% BR_SUBSET_FROM_IDX  Builds a BreachTraceSystem containing only traces idx.
Br_sub = BreachTraceSystem(sig_names);
for ii = 1:numel(idx)
    Br_sub.AddTrace(TR{idx(ii)});
end
end
