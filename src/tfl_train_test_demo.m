function tfl_train_test_demo
% TFL TRAIN/TEST DEMO
% - Generates synthetic traces (normal + occasional ringing bursts)
% - Builds z_out(t): STFT energy outside a user-specified "safe" frequency band
% - Uses Breach to monitor STL formula:   G_[0,T] (zout[t] <= th)
% - Trains on ALL training traces, tuning 'th' via simulated annealing to MINIMIZE training MCR
% - Evaluates on a held-out test set; prints metrics and makes overview plots
%
% Requirements:
%   - MATLAB R2019b+ recommended
%   - Global Optimization Toolbox (simulannealbnd)
%   - Breach toolbox on MATLAB path (BreachTraceSystem, STL_Formula, ...)

%% 0) Setup & config
rng(128);

% Sampling
Fs   = 200;                       % Hz
T    = 20;                        % seconds per trace
M    = round(Fs*T);               % samples per trace
t    = (0:M-1)'/Fs;
fmax = 15;                        % Hz cutoff for TF plots/aggregation

% Dataset
n_traces       = 512;             % number of traces
ring_frac      = 0.05;            % fraction with injected ringing
f0_sim         = 1.25;              % ringing freq used in SIMULATION (unknown to learner)
train_ratio    = 0.8;             % 80/20 train/test split (stratified)
safe_bands     = [0.0 1.0];       % rows [fmin fmax] defining NORMAL energy bands
A_ring         = 0.10;            % ringing amplitude
dur_ring_range = [0.8 2.0];       % s
SNR_dB         = 15;              % baseline SNR for normals

% STFT (choose longer window for sub-1 Hz structure)
win_len  = round(4*Fs);
noverlap = round(0.9*win_len);
nfft     = 2^nextpow2(win_len);

% Aggregation across out-of-safe freqs
agg_mode = 'mean';                % 'mean' or 'max'

% Simulated Annealing
init_temp = 100.0;
max_iter = 16;

% Figure saver: create figs/<yyyy-MM-dd_HHmmss>/ and reset counter
run_id  = string(datetime('now'), 'yyyy-MM-dd_HHmmss');
run_dir = fullfile('figs', run_id);
savefig_seq('init', run_dir);
fprintf('Saving figures to: %s\n', run_dir);


%% 1) Build the dataset (X, labels)
[X, labels] = build_dataset(Fs, T, n_traces, ring_frac, f0_sim, A_ring, dur_ring_range, SNR_dB);

%% 2) Compute z_out(t) via STFT
Z = compute_zout_from_stft(X, Fs, win_len, noverlap, nfft, safe_bands, fmax, agg_mode);

%% 3) Quick sample plots
pick_norm = find(labels==0, 2, 'first');
pick_ring = find(labels==1, 2, 'first');
picks = [pick_norm(:); pick_ring(:)];
figure('Name','Sample time series'); tiledlayout(numel(picks),1,'Padding','compact','TileSpacing','compact');
for i = 1:numel(picks)
    k = picks(i); nexttile; plot(t, X(:,k)); grid on;
    title(sprintf('Trace %d (%s)', k, tern(labels(k)==0,'normal','ringing')));
    xlabel('t [s]'); ylabel('x(t)');
end
savefig_seq('save', gcf, 'sample_time_series');

%% 4) STFT transform energy plots
figure('Name','STFT power + z_{out}(t)');
tiledlayout(numel(picks),2,'Padding','compact','TileSpacing','compact');
for i = 1:numel(picks)
    k = picks(i);

    [S,F,Tstft] = spectrogram(X(:,k), win_len, noverlap, nfft, Fs, 'yaxis');
    P = abs(S).^2;

    keep = F <= fmax;
    F    = F(keep);
    P    = P(keep,:);

    nexttile; imagesc(Tstft, F, 10*log10(P+eps)); axis xy; colorbar;
    ylim([0 fmax]);    % show only up to cutoff
    hold on; yline(safe_bands,'w--'); hold off;
    xlabel('t [s]'); ylabel('f [Hz]'); title(sprintf('Trace %d STFT (dB)', k));
    nexttile; plot(t, Z(:,k)); grid on; xlabel('t [s]'); ylabel('z_{out}(t)');
    title(sprintf('Trace %d out-of-safe energy', k));
end
savefig_seq('save', gcf, 'stft_and_zout_examples');

%% 5) Breach traces and STL formula
if ~exist('BreachTraceSystem','class')
    error('Breach not found. Add Breach to the MATLAB path before running.');
end
[Br, TR] = make_breach_traces(t, X, Z, {'x','zout'});

phi_str = 'alw_[0,T] ( zout[t] <= th )';
phi     = STL_Formula('phi', phi_str);
phi     = set_params(phi, {'T'}, T);

%% 6) Stratified train/test split (train on ALL training traces)
[idx_train, idx_test] = split_train_test(labels, train_ratio);
Br_train = br_subset_from_idx(TR, {'x','zout'}, idx_train);
Br_test  = br_subset_from_idx(TR, {'x','zout'}, idx_test);

ytrue_train = 1 - labels(idx_train);   % 1=normal, 0=ringing

% Data-driven prior interval for theta from TRAIN set
th_lo = min(Z(:,idx_train), [], 'all');
th_hi = max(Z(:,idx_train), [], 'all');
theta0 = 0.5*(th_lo+th_hi);

%% 7) Simulated annealing (Global Opt. Toolbox) over theta to MINIMIZE training MCR
sa_obj  = @(theta) mcr_on_split(Br_train, phi, theta, ytrue_train);
opts_sa = optimoptions('simulannealbnd', ...
    'Display', 'iter', ...
    'InitialTemperature', init_temp, ...
    'MaxIterations', max_iter, ...
    'PlotFcn', {@saplotbestf, @saplotf, @saplottemperature});

[theta_star, mcr_train_best, exitflag, output] = simulannealbnd(sa_obj, theta0, th_lo, th_hi, opts_sa);
fprintf('TRAIN SA: theta* = %.6g | MCR_train = %.4f | prior box = [%.6g, %.6g] | exitflag=%d, iters=%d\n', ...
    theta_star, mcr_train_best, th_lo, th_hi, exitflag, output.iterations);

%% 8) Final evaluation on TEST set
Br_test.SetParamSpec('th', theta_star);
rob_test   = Br_test.CheckSpec(phi);
yhat_test  = double(rob_test>=0);       % 1=normal, 0=ringing
ytrue_test  = 1 - labels(idx_test);     % 1=normal, 0=ringing

C = confusionmat(ytrue_test, yhat_test);
TN = C(1,1); FP = C(1,2); FN = C(2,1); TP = C(2,2);
N  = sum(C,'all');

acc = (TP + TN) / N;
mcr = 1 - acc;
prec = TP / max(TP + FP, 1);
tpr  = TP / max(TP + FN, 1);
fpr  = FP / max(FP + TN, 1);

fprintf('\nTEST PERFORMANCE\n')
fprintf('                Pred 0        Pred 1\n');
fprintf('Actual 0        %6d        %6d\n', TN, FP);
fprintf('Actual 1        %6d        %6d\n', FN, TP);
fprintf('acc=%.3f  mcr=%.3f  prec=%.3f  tpr=%.3f  fpr=%.3f  | n_test=%d\n\n', ...
    acc, mcr, prec, tpr, fpr, N);

%% 9) Full-dataset overview with final theta*
Br_eval = Br.copy();
Br_eval.SetParamSpec('th', theta_star);
rob_total   = Br_eval.CheckSpec(phi);
yhat_total  = double(rob_total>=0);
ytrue_total = 1 - labels;

mis_idx  = find(yhat_total~=ytrue_total);
corr_idx = find(yhat_total==ytrue_total);
n_mis    = numel(mis_idx);

% One normal and one ringing correctly classified (if exist)
exN = find(ytrue_total==1 & yhat_total==1, 1, 'first');
exR = find(ytrue_total==0 & yhat_total==0, 1, 'first');

if ~isempty(exN) && ~isempty(exR)
    figure('Name','Example normal vs ringing with \theta^*');
    subplot(2,2,1); plot(t, X(:,exN)); grid on; title('x(t): normal'); xlabel('t [s]');
    subplot(2,2,3); plot(t, Z(:,exN)); hold on; yline(theta_star,'r--','\theta^*'); grid on;
    title('z_{out}(t): normal'); xlabel('t [s]');
    subplot(2,2,2); plot(t, X(:,exR)); grid on; title('x(t): ringing'); xlabel('t [s]');
    subplot(2,2,4); plot(t, Z(:,exR)); hold on; yline(theta_star,'r--','\theta^*'); grid on;
    title('z_{out}(t): ringing'); xlabel('t [s]');
    savefig_seq('save', gcf, 'example_normal_vs_ringing');
end

figure('Name','ALL x(t): misclassified highlighted','Color','w'); hold on; grid on;
h_corr = plot(t, X(:,corr_idx), 'Color', [0.7 0.7 0.7]);
if ~isempty(mis_idx)
    h_mis = plot(t, X(:,mis_idx), 'Color', 'r', 'LineWidth', 1.2);
end
xlabel('t [s]'); ylabel('x(t)');
title(sprintf('All traces x(t), Misclassified: %d / %d', n_mis, size(X,2)));
if isempty(mis_idx)
    legend(h_corr(1), {'correct'}, 'Location','northwest');
else
    legend([h_corr(1), h_mis(1)], {'correct','misclassified'}, 'Location','northwest');
end
savefig_seq('save', gcf, 'all_x_misclassified');

% OLD PLOTTING FUNCTION
% figure('Name','ALL x(t): misclassified highlighted','Color','w');
% plot(t, X(:,corr_idx), 'Color', [0.8 0.8 0.8]); hold on; grid on;
% if ~isempty(mis_idx)
%     h_mis = plot(t, X(:,mis_idx), 'LineWidth', 1.2);
%     set(h_mis, 'Color', 'r');
% end
% xlabel('t [s]'); ylabel('x(t)');
% title(sprintf('All traces x(t), Misclassified: %d / %d', n_mis, size(X,2)));
% legend({'correct','misclassified'}, 'Location','northwest');
% savefig_seq('save', gcf, 'all_x_misclassified');

figure('Name','ALL z_{out}(t): misclassified highlighted','Color','w'); hold on; grid on;
h_corr = plot(t, Z(:,corr_idx), 'Color', [0.7 0.7 0.7]);
h_mis = plot(t, Z(:,mis_idx), 'Color', 'r', 'LineWidth', 1.2);
h_thr = yline(theta_star, '--k', '\theta^*', 'LineWidth', 1.4);
xlabel('t [s]'); ylabel('z_{out}(t)');
title(sprintf('All traces z_{out}(t) — Misclassified: %d / %d', n_mis, size(Z,2)));
legend([h_corr(1), h_mis(1), h_thr], ...
       {'correct','misclassified','\theta^*'}, ...
       'Location','northwest');
savefig_seq('save', gcf, 'all_zout_misclassified');

% OLD PLOTTING FUNCTION
% figure('Name','ALL z_{out}(t): misclassified highlighted','Color','w');
% plot(t, Z(:,corr_idx), 'Color', [0.8 0.8 0.8]); hold on; grid on;
% if ~isempty(mis_idx)
%     h_mis2 = plot(t, Z(:,mis_idx), 'LineWidth', 1.2);
%     set(h_mis2, 'Color', [0.85 0.33 0.10]);
% end
% yline(theta_star, '--', '\theta^*', 'LineWidth', 1.4);
% xlabel('t [s]'); ylabel('z_{out}(t)');
% title(sprintf('All traces z_{out}(t) — Misclassified: %d / %d', n_mis, size(Z,2)));
% legend({'correct','misclassified','\theta^*'}, 'Location','northwest');
% savefig_seq('save', gcf, 'all_zout_misclassified');

figure('Name','Robustness per trace','Color','w'); hold on; grid on;
idx_norm_all = find(ytrue_total==1);
idx_ring_all = find(ytrue_total==0);
h_norm = scatter(idx_norm_all, rob_total(idx_norm_all), 18, 'filled');
h_ring = scatter(idx_ring_all, rob_total(idx_ring_all), 18, 'r', 'filled');
if ~isempty(mis_idx)
    h_mis = scatter(mis_idx, rob_total(mis_idx), 60, 'o', ...
        'MarkerEdgeColor','k', 'LineWidth',1.2, 'MarkerFaceColor','none');
end
h_zero = yline(0, 'k--');
xlabel('trace id'); ylabel('robustness r_\phi');
title('Robustness by trace  (normal=blue, ringing=orange, misclassified=black circle)');
if isempty(mis_idx)
    legend([h_norm, h_ring, h_zero], {'normal','ringing','r=0'}, 'Location','best');
else
    legend([h_norm, h_ring, h_mis, h_zero], {'normal','ringing','misclassified','r=0'}, 'Location','best');
end
savefig_seq('save', gcf, 'robustness_by_trace');

% OLD PLOTTING FUNCTION
% figure('Name','Robustness per trace','Color','w');
% scatter(1:size(Z,2), rob_total, 18, (ytrue_total==1), 'filled'); hold on; grid on;
% if ~isempty(mis_idx), scatter(mis_idx, rob_total(mis_idx), 50, 'r', 'LineWidth', 1.2); end
% yline(0, 'k--');
% xlabel('trace id'); ylabel('robustness r_\phi');
% title('Robustness by trace  (color: normal=1 / ringing=0; red circle = misclassified)');
% colormap([0.85 0.33 0.10; 0 0.45 0.74]); colorbar('Ticks',[0 1],'TickLabels',{'ringing','normal'});
% savefig_seq('save', gcf, 'robustness_by_trace');

end
