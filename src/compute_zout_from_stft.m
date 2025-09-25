function Z = compute_zout_from_stft(X, Fs, win_len, noverlap, nfft, safe_bands, fmax, agg_mode)
% COMPUTE_ZOUT_FROM_STFT  Builds z_out(t) by aggregating energy outside safe bands.
% Returns Z: M x n_traces on the original time grid.
[M, n_traces] = size(X);
t = (0:M-1)'/Fs;

z_all = cell(1, n_traces);
for k = 1:n_traces
    [S,F,Tstft] = spectrogram(X(:,k), win_len, noverlap, nfft, Fs, "yaxis");
    P = abs(S).^2;

    keep = F <= fmax;
    F    = F(keep);
    P    = P(keep,:);

    in_safe = false(size(F));
    for b = 1:size(safe_bands,1)
        in_safe = in_safe | (F>=safe_bands(b,1) & F<=safe_bands(b,2));
    end
    out_mask = ~in_safe;

    switch lower(agg_mode)
        case 'max',  z = max(P(out_mask, :), [], 1);
        case 'mean', z = mean(P(out_mask, :), 1);
        otherwise, error('Unknown agg_mode: %s', agg_mode);
    end
    z_all{k} = struct('Tstft', Tstft(:), 'z', z(:));
end

Z = zeros(M, n_traces);
for k = 1:n_traces
    Z(:,k) = interp1(z_all{k}.Tstft, z_all{k}.z, t, 'nearest', 'extrap');
end
end
