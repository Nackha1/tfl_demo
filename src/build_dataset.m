function [X, labels] = build_dataset(Fs, T, n_traces, ring_frac, f0, A_ring, dur_range, SNR_dB)
% BUILD_DATASET  Synthetic normals + ringing bursts
M = round(Fs*T);
t = (0:M-1)'/Fs;

X = zeros(M, n_traces);
labels = zeros(1, n_traces); % 0=normal, 1=ringing

% base normals
for k = 1:n_traces
    base  = 0.5*sin(2*pi*0.4*t + 2*pi*rand) + 0.2*sin(2*pi*0.9*t + 2*pi*rand);
    noise = randn(size(t)); noise = noise/std(noise);
    sigma = rms(base)/db2mag(SNR_dB);
    X(:,k) = base + sigma*noise;
end

% pick ringing traces
idx_ring = randperm(n_traces, round(ring_frac*n_traces));
labels(idx_ring) = 1;

% inject ringing
for k = idx_ring
    dur = rand*(diff(dur_range)) + dur_range(1);
    ts  = rand*(T - dur); te = ts + dur;
    seg = (t>=ts) & (t<=te);
    L   = sum(seg); w = zeros(M,1);
    w(seg) = tukeywin(L, 0.5);
    X(:,k) = X(:,k) + A_ring*w.*sin(2*pi*f0*t + 2*pi*rand);
end
end
