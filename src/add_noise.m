function X_noisy = add_noise(X_clean, SNR_dB)
% ADD_NOISE_SNR  Adds white Gaussian noise to match target SNR [dB].
%
%   X_noisy = add_noise(X_clean, SNR_dB)
%
% INPUTS:
%   X_clean : (M x N) matrix of clean signals
%   SNR_dB  : Target signal-to-noise ratio (in decibels)
%
% OUTPUT:
%   X_noisy : (M x N) noisy signals
%
% The noise power is scaled so that:
%   SNR = 10*log10( signal_power / noise_power )

[M, N] = size(X_clean);
X_noisy = zeros(M, N);

for k = 1:N
    x = X_clean(:,k);
    noise = randn(size(x));
    noise = noise / std(noise);
    sigma = rms(x) / db2mag(SNR_dB);
    X_noisy(:,k) = x + sigma * noise;
end
end
