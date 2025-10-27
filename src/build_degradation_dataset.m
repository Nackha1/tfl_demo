function [X, labels] = build_degradation_dataset(Fs, T, n_traces, degr_frac, f0, f1_range, A0)
% BUILD_DEGRADATION_DATASET  Synthetic dataset of normal and degrading oscillators.
%
%   [X, labels, f_inst] = build_degradation_dataset(Fs, T, n_traces, degr_frac, f0, f1_range, A0)
%
% Simulates a population of oscillatory machines, where a fraction shows
% gradual frequency degradation (slowing down) over time.
%
% INPUTS:
%   Fs         : Sampling frequency [Hz]
%   T          : Duration per trace [s]
%   n_traces   : Total number of time series to generate
%   degr_frac  : Fraction of degrading traces (0–1)
%   f0         : Initial oscillation frequency [Hz]
%   f1_range   : [f_min, f_max] final frequency range for degraded traces
%   A0         : Initial amplitude (scalar)
%
% OUTPUTS:
%   X        : (M x n_traces) matrix of generated signals
%   labels   : (1 x n_traces)  0 = normal, 1 = degrading
%

%% --- Precompute parameters
M = round(Fs * T);
t = (0:M-1)' / Fs;

X = zeros(M, n_traces);
labels = zeros(1, n_traces);  % 0 = normal, 1 = degrading

%% --- Generate all traces
for k = 1:n_traces
    % Add a bit of variability in base frequency even for normals
    f_base = f0 * (0.9 + 0.2*rand);  % +-10% variation

    % Default: no degradation
    f_t = f_base * ones(size(t));
    A_t = A0 * ones(size(t));

    % Pick degrading traces
    if rand < degr_frac
        labels(k) = 1;
        f_end = rand*(f1_range(2)-f1_range(1)) + f1_range(1);
        % Linear decay of frequency over time
        f_t = linspace(f_base, f_end, M)';
        % Optional amplitude decay (exponential)
        A_t = A0 * exp(-0.3 * (t/T));  % mild amplitude loss
    end

    % Integrate instantaneous frequency to get phase
    phase = 2*pi*cumtrapz(t, f_t);
    X(:,k)  = A_t .* sin(phase + 2*pi*rand);
end
end