function Z = compute_zout_from_cwt(X, Fs, safe_bands, fmax, agg_mode)
% COMPUTE_ZOUT_FROM_CWT  Builds z_out(t) by aggregating energy outside safe bands,
% ignoring coefficients inside the cone of influence (COI).
%
% INPUTS:
%   X           : M x n_traces matrix of time-domain signals
%   Fs          : sampling frequency [Hz]
%   safe_bands  : Nx2 matrix of [fmin fmax] ranges considered "safe"
%   fmax        : maximum frequency to include in the analysis
%   agg_mode    : 'mean' or 'sum' for aggregation method
%
% OUTPUT:
%   Z           : M x n_traces aggregated indicator on the original time grid

    [M, n_traces] = size(X);

    fb = cwtfilterbank(SignalLength=M, SamplingFrequency=Fs);

    Z = zeros(M, n_traces);
    for k = 1:n_traces
        % --- Continuous Wavelet Transform
        [S, F, coi] = cwt(X(:,k), FilterBank=fb);
        P = abs(S);          % magnitude (or use abs(S).^2 for energy)
        % keep = F <= fmax;    % limit frequency band
        % 
        % F = F(keep);
        % P = P(keep, :);

        % --- Ignore frequencies inside the cone of influence
        for t_idx = 1:length(coi)
            inside_coi = F < coi(t_idx);  % frequencies below the limit are unreliable
            % P(inside_coi, t_idx) = NaN;
        end

        % plot(t,coi);

        % --- Determine frequencies outside safe bands
        in_safe = false(size(F));
        for b = 1:size(safe_bands, 1)
            in_safe = in_safe | (F >= safe_bands(b,1) & F <= safe_bands(b,2));
        end
        out_mask = ~in_safe;

        % out_mask = true(size(F));

        % --- Aggregate energy outside safe bands, ignoring NaNs (COI)
        switch lower(agg_mode)
            case 'mean'
                z = mean(P(out_mask, :), 1, 'omitmissing');
            case 'sum'
                z = sum(P(out_mask, :), 1, 'omitmissing');
            case 'max_f'
                % z = max(P(out_mask, :), [], 1, 'omitmissing');
                % Find the frequency with maximum amplitude at each time
                [~, idx_max] = max(P(out_mask, :), [], 1, 'omitmissing');
                z = F(idx_max);
            otherwise
                error('Unknown agg_mode: %s', agg_mode);
        end

        z(isnan(z)) = 0;
        % --- Store result
        Z(:,k) = z(:);
    end
end