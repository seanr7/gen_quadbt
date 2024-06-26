%% RUNME
% Script file to run all experiments in ...

%
% This file is part of the archive Code and Results for Numerical 
% Experiments in "..."
% Copyright (c) 2024 Sean Reiter, Steffen W. R. Werner, and ...
% All rights reserved.
% License: BSD 2-Clause license (see COPYING)
%

clc;
clear all;
close all;

% Get and set all paths
[rootpath, filename, ~] = fileparts(mfilename('fullpath'));
loadname            = [rootpath filesep() ...
    'data' filesep() filename];
savename            = [rootpath filesep() ...
    'results' filesep() filename];

% Add paths to drivers and data
addpath([rootpath, '/drivers'])
addpath([rootpath, '/data'])

% Write .log file, put in `out' folder
if exist([savename '.log'], 'file') == 2
    delete([savename '.log']);
end
outname = [savename '.log']';

diary(outname)
diary on; 

fprintf(1, ['SCRIPT: ' upper(filename) '\n']);
fprintf(1, ['========' repmat('=', 1, length(filename)) '\n']);
fprintf(1, '\n');


%% Load data.
fprintf(1, 'Loading benchmark problem.\n')
fprintf(1, '--------------------------\n')

% Load circuit model
n               = 200;
[A, B, C, D, ~] = circuit(n);  

% Load ISS SLICOT benchmark
% load('iss.mat')
% % Conver matrices from sparse to full
% A      = full(A);    
% B      = full(B);    
% C      = full(C);
% p      = 3;  
% [n, ~] = size(A);
% % Uncomment the lines below if you want to make the problem single-input,
% % single-output
% % B = B(:, 1);    
% % C = C(1, :); 
% % p = 1;
% % Some nontrivial input feedthrough is required, to do self-weighted fwbt
% eps = .1;
% D   = eps*eye(p, p);

%% Frequency-limited balanced truncation from data.
fprintf(1, 'Building frequency-limited quadbt reduced model.\n')
fprintf(1, '------------------------------------------------\n')

% Instantiate sampler class
% In frequency-limited quadbt (flquadbt) the generic (quadbt) sampler class
% is used. The `frequency-limited' aspect of the approximation is captured
% by the limited frequency range wherein the transfer function is sampled.
% No D term required for flquadbt.

qflbt_sampler = QuadBTSampler(A, B, C, D);

% Note that the dominant frequency character of the problem is in the range
% from i[1e-1, 1e4]
a     = -1;
b     = 1;
omega = [10^a, 10^b]; % Limited frequency band of interest
N     = 160;          % Number of quadrature nodes

% Prepare quadratute weights, nodes
[nodesl, weightsl, nodesr, weightsr] = trap_rule([a, b], N, true);

% Instantiate reductor class
qflbt_engine = GeneralizedQuadBTReductor(qflbt_sampler, nodesl, nodesr, ...
    weightsl, weightsr);

% Verify the build of the Loewner matrices is correct
checkloewner = true;
if checkloewner == true
    % Loewner quadruple for quadflbt
    Lbar = qflbt_engine.Lbar;    Mbar = qflbt_engine.Mbar;    
    Hbar = qflbt_engine.Hbar;    Gbar = qflbt_engine.Gbar; 

    % Quadrature based square root factors for comparison
    Ux = qflbt_sampler.right_sqrt_factor(nodesr, weightsr);
    Ly = qflbt_sampler.left_sqrt_factor(nodesl, weightsl);

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Lbar: Error || Lbar - Ly.T * Ux     ||: %.16f\n', norm(Ly' * Ux - Lbar, 2))
    fprintf('Check for Mbar: Error || Mbar - Ly.T * A * Ux ||: %.16f\n', norm(Ly' * A * Ux - Mbar, 2))
    fprintf('Check for Hbar: Error || Hbar - Ly.T * B      ||: %.16f\n', norm(Ly' * B - Hbar, 2))
    fprintf('Check for Gbar: Error || Gbar - C * Ux        ||: %.16f\n', norm(C * Ux - Gbar, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
end

% Specify order of reduction
r = 20;

% Construct system matrices from data for quadflbt rom
[Ar_qflbt, Br_qflbt, Cr_qflbt] = qflbt_engine.reduce(r);
Dr_qflbt                       = D;
qflbt_rom                      = ss(Ar_qflbt, Br_qflbt, Cr_qflbt, Dr_qflbt);

% Create benchmark intrusive model
opts                  = ml_morlabopts('ml_ct_d_ss_bt');
opts.Order            = r;
opts.OrderComputation = 'Order';
opts.FreqRange        = omega;
fom                   = ss(A, B, C, D);
[flbt_rom , info]     = ml_ct_d_ss_flbt(fom, opts);
Ar_flbt               = flbt_rom.A;
Br_flbt               = flbt_rom.B;
Cr_flbt               = flbt_rom.C;
Dr_flbt               = D;

%% Plot frequency response functions.
lens        = 750;
% s           = linspace(0, 2*pi*250, lens);
s           = logspace(-1, 4, lens);
res_fom     = zeros(lens, 1);
res_r_flbt  = zeros(lens, 1);
res_r_qflbt = zeros(lens, 1);
err_r_flbt  = zeros(lens, 1);
err_r_qflbt = zeros(lens, 1);
I           = eye(n, n);
Ir          = eye(r, r);

% Plot frequency response along imaginary axis
for ii=1:lens
    fprintf(1, 'Frequency step %d, s=1i*%.2f ...\n ', ii, s(ii))
    % Evaluate transfer function magnitude
    Gfo             = C*((1i*s(ii)*I - A)\B) + D;
    Gr_flbt         = Cr_flbt*((1i*s(ii)*Ir - Ar_flbt)\Br_flbt) + Dr_flbt;
    Gr_qflbt        = Cr_qflbt*((1i*s(ii)*Ir - Ar_qflbt)\Br_qflbt) + Dr_qflbt;
    res_fom(ii)     = norm(Gfo, 2); 
    res_r_flbt(ii)  = norm(Gr_flbt, 2); 
    res_r_qflbt(ii) = norm(Gr_qflbt, 2); 
    err_r_flbt(ii)  = norm(Gfo - Gr_flbt, 2); 
    err_r_qflbt(ii) = norm(Gfo - Gr_qflbt, 2); 
    fprintf(1, '----------------------------------------------------------------------\n');
end


% Plot colors
ColMat = zeros(6,3);
ColMat(1,:) = [ 0.8500    0.3250    0.0980];
ColMat(2,:) = [0.3010    0.7450    0.9330];
ColMat(3,:) = [  0.9290    0.6940    0.1250];
ColMat(4,:) = [0.4660    0.6740    0.1880];
ColMat(5,:) = [0.4940    0.1840    0.5560];
ColMat(6,:) = [1 0.4 0.6];

figure(1)
fs = 12;
% Magnitudes
set(gca, 'fontsize', 10)
subplot(2,1,1)
loglog(s, res_fom, '-', 'linewidth', 2, 'color', ColMat(1,:)); hold on
loglog(s, res_r_flbt, '-.', 'linewidth', 2, 'color', ColMat(2,:)); 
loglog(s, res_r_qflbt, '--', 'linewidth', 2, 'color', ColMat(3,:)); 
leg = legend('Full-order', 'flbt', 'qflbt', 'location', 'southeast', ...
    'orientation', 'horizontal', 'interpreter', 'latex');
set(leg, 'fontsize', 10, 'interpreter', 'latex')
xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')

% Relative errors
subplot(2,1,2)
loglog(s, err_r_flbt./res_fom, '-.', 'linewidth', 2, 'color', ColMat(2,:));
hold on
loglog(s, err_r_qflbt./res_fom, '--', 'linewidth', 2, 'color', ColMat(3,:)); 
leg = legend('flbt', 'qflbt', 'location', 'southeast', ...
    'orientation', 'horizontal', 'interpreter', 'latex');
set(leg, 'fontsize', 10, 'interpreter', 'latex')
xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')

print -depsc2 results/qflbt_mag_err


%% Frequency-weighted bt with one-sided self weights.
% Self weighted with Wo(s) = G(s)^{-1} and Wi(s) = Im
fprintf(1, 'Building self-weighted quadbt reduced model.\n')
fprintf(1, '--------------------------------------------\n')

% First, realization of G(s)^{-1} is needed to sample it
Dinv    = 1/D;
Ainv    = A - B * Dinv * C;
Binv    = -B * Dinv;
Cinv    = Dinv * C;
inv_fom = ss(Ainv, Binv, Cinv, Dinv);

qfwbt_sampler = QuadFWBTSampler(A, B, C, D, zeros(n, n), zeros(n, 1), zeros(1, n), ...
    1, Ainv, Binv, Cinv, Dinv);

% Note that the dominant frequency character of the problem is in the range
% from i[1e-1, 1e4]
% Look at holw range now
a     = -1;
b     = 4;
N     = 80;          % Number of quadrature nodes

% Prepare quadratute weights, nodes
[nodesl, weightsl, nodesr, weightsr] = trap_rule([a, b], N, true);

% Instantiate reductor class
qfwbt_engine = GeneralizedQuadBTReductor(qflbt_sampler, nodesl, nodesr, ...
    weightsl, weightsr);

% Verify the build of the Loewner matrices is correct
checkloewner = true;
if checkloewner == true
    % Loewner quadruple for quadflbt
    Lbar = qfwbt_engine.Lbar;    Mbar = qfwbt_engine.Mbar;    
    Hbar = qfwbt_engine.Hbar;    Gbar = qfwbt_engine.Gbar; 

    % Quadrature based square root factors for comparison
    Ux = qfwbt_sampler.right_sqrt_factor(nodesr, weightsr);
    Ly = qfwbt_sampler.left_sqrt_factor(nodesl, weightsl);

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Lbar: Error || Lbar - Ly.T * Ux     ||: %.16f\n', norm(Ly' * Ux - Lbar, 2))
    fprintf('Check for Mbar: Error || Mbar - Ly.T * A * Ux ||: %.16f\n', norm(Ly' * A * Ux - Mbar, 2))
    fprintf('Check for Hbar: Error || Hbar - Ly.T * B      ||: %.16f\n', norm(Ly' * B - Hbar, 2))
    fprintf('Check for Gbar: Error || Gbar - C * Ux        ||: %.16f\n', norm(C * Ux - Gbar, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
end

r = 20;
[Ar_qfwbt, Br_qfwbt, Cr_qfwbt] = qfwbt_engine.reduce(r);
Dr_qfwbt                       = D;
qfwbt_rom                      = ss(Ar_qfwbt, Br_qfwbt, Cr_qfwbt, D);


%% Plot frequency response functions.
lens           = 750;
s              = logspace(-1, 4, lens);
res_fom        = zeros(lens, 1);
res_r_qfwbt    = zeros(lens, 1);
relerr_r_qfwbt = zeros(lens, 1);
I              = eye(n, n);
Ir             = eye(r, r);

% Plot frequency response along imaginary axis
for ii=1:lens
    fprintf(1, 'Frequency step %d, s=1i*%.2f ...\n ', ii, s(ii))
    % Evaluate transfer function magnitude
    Gfo                = C*((1i*s(ii)*I - A)\B) + D;
    Gr_qfwbt           = Cr_qfwbt*((1i*s(ii)*Ir - Ar_qfwbt)\Br_qfwbt) + Dr_qfwbt;
    res_fom(ii)        = norm(Gfo, 2); 
    res_r_qfwbt(ii)    = norm(Gr_qfwbt, 2); 
    relerr_r_qfwbt(ii) = norm((Gfo\(Gfo - Gr_qfwbt)), 2); 
    fprintf(1, '----------------------------------------------------------------------\n');
end

figure(2)
fs = 12;
% Magnitudes
set(gca, 'fontsize', 10)
subplot(2,1,1)
loglog(s, res_fom, '-', 'linewidth', 2, 'color', ColMat(1,:)); hold on
loglog(s, res_r_qfwbt, '--', 'linewidth', 2, 'color', ColMat(2,:)); 
leg = legend('Full-order', 'qfwbt', 'location', 'southeast', ...
    'orientation', 'horizontal', 'interpreter', 'latex');
set(leg, 'fontsize', 10, 'interpreter', 'latex')
xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')

% Relative errors
subplot(2,1,2)
loglog(s, relerr_r_qfwbt, '--', 'linewidth', 2, 'color', ColMat(3,:)); 
leg = legend('qfwbt', 'location', 'southeast', ...
    'orientation', 'horizontal', 'interpreter', 'latex');
set(leg, 'fontsize', 10, 'interpreter', 'latex')
xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
ylabel('$||\mathbf{G}(s)^{-1}(\mathbf{G}(s)-\mathbf{G}_{r}(s))||_2$', 'fontsize', fs, 'interpreter', 'latex')

print -depsc2 results/qfwbt_mag_err


%% Frequency-limited bt with modified Gramians.
fprintf(1, 'Now, using modified Gramians.\n')
fprintf(1, '-----------------------------\n')

% Note that the dominant frequency character of the problem is in the range
% from i[1e-1, 1e4]
a     = -1;
b     = 1;
omega = [10^a, 10^b]; % Limited frequency band of interest
N     = 160;          % Number of quadrature nodes

% Prepare quadratute weights, nodes
[nodesl, weightsl, ~, ~] = trap_rule([a, b], N, true);  % Band limited
[~, ~, nodesr, weightsr] = trap_rule([-1, 4], N, true); % Full band

% Instantiate reductor class; just apply different sets of nodes
% Left nodes/weights used in approximating obsv Gramian, right for reach
% So, the observability Gramian is the limited one here...
quadmodflbt_engine = GeneralizedQuadBTReductor(qflbt_sampler, nodesl, nodesr, ...
    weightsl, weightsr);

% Verify the build of the Loewner matrices is correct
checkloewner = true;
if checkloewner == true
    % Loewner quadruple for quadflbt
    Lbar = quadmodflbt_engine.Lbar;    Mbar = quadmodflbt_engine.Mbar;    
    Hbar = quadmodflbt_engine.Hbar;    Gbar = quadmodflbt_engine.Gbar; 

    % Quadrature based square root factors for comparison
    Ux = qflbt_sampler.right_sqrt_factor(nodesr, weightsr);
    Ly = qflbt_sampler.left_sqrt_factor(nodesl, weightsl);

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Lbar: Error || Lbar - Ly.T * Ux     ||: %.16f\n', norm(Ly' * Ux - Lbar, 2))
    fprintf('Check for Mbar: Error || Mbar - Ly.T * A * Ux ||: %.16f\n', norm(Ly' * A * Ux - Mbar, 2))
    fprintf('Check for Hbar: Error || Hbar - Ly.T * B      ||: %.16f\n', norm(Ly' * B - Hbar, 2))
    fprintf('Check for Gbar: Error || Gbar - C * Ux        ||: %.16f\n', norm(C * Ux - Gbar, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
end

% Specify order of reduction
r = 20;

% Construct system matrices from data for quadflbt rom
[Ar_qmodflbt, Br_qmodflbt, Cr_qmodflbt] = quadmodflbt_engine.reduce(r);
Dr_qmodflbt                             = D;
qmodflbt_rom                            = ss(Ar_qmodflbt, Br_qmodflbt, Cr_qmodflbt, ...
                                               Dr_qmodflbt);

% Get one Gramian from true BT, one from FL
% First, fl Gramians
flopts                  = ml_morlabopts('ml_ct_d_ss_flbt');
flopts.Order            = r;
flopts.OrderComputation = 'Order';
flopts.FreqRange        = omega;
flopts.StoreGramians    = 1;
[~, flinfo]             = ml_ct_d_ss_flbt(fom, flopts);
Lobsv_fl                = flinfo.GramFacO;

% Then, Lyapunov Gramians
opts                  = ml_morlabopts('ml_ct_d_ss_flbt');
opts.Order            = r;
opts.OrderComputation = 'Order';
opts.StoreGramians    = 1;
[~, info]             = ml_ct_d_ss_bt(fom, opts);
Ureach                = info.GramFacC;

[Z, S, Y] = svd(Ureach'*Lobsv_fl);

% Compute projection matrices
V = Ureach*Z(:, 1:r)*S(1:r, 1:r)^(-1/2);   % Right
W = Lobsv_fl*Y(:, 1:r)*S(1:r, 1:r)^(-1/2); % Left
% Now, implement the modified BT
Ar_modflbt = W'*A*V;
Br_modflbt = W'*B;
Cr_modflbt = C*V;
Dr_modflbt = D;


%% Plot frequency response functions.
lens           = 500;
s              = logspace(-1, 4, lens);
res_fom        = zeros(lens, 1);
res_r_modflbt  = zeros(lens, 1);
res_r_qmodflbt = zeros(lens, 1);
err_r_modflbt  = zeros(lens, 1);
err_r_qmodflbt = zeros(lens, 1);
I              = eye(n, n);
Ir             = eye(r, r);
 
% Plot frequency response along imaginary axis
for ii=1:lens
    fprintf(1, 'Frequency step %d, s=1i*%.2f ...\n ', ii, s(ii))
    % Evaluate transfer function magnitude
    Gfo                = C*((1i*s(ii)*I - A)\B) + D;
    Gr_modflbt         = Cr_modflbt*((1i*s(ii)*Ir - Ar_modflbt)\Br_modflbt) + Dr_modflbt;
    Gr_qmodflbt        = Cr_qmodflbt*((1i*s(ii)*Ir - Ar_qmodflbt)\Br_qmodflbt) + Dr_qmodflbt;
    res_fom(ii)        = norm(Gfo, 2); 
    res_r_modflbt(ii)  = norm(Gr_modflbt, 2); 
    res_r_qmodflbt(ii) = norm(Gr_qmodflbt, 2); 
    err_r_modflbt(ii)  = norm(Gfo - Gr_modflbt, 2); 
    err_r_qmodflbt(ii) = norm(Gfo - Gr_qmodflbt, 2); 
    fprintf(1, '----------------------------------------------------------------------\n');
end

figure(3)
fs = 12;
% Magnitudes
set(gca, 'fontsize', 10)
subplot(2,1,1)
loglog(s, res_fom, '-', 'linewidth', 2, 'color', ColMat(1,:)); hold on
loglog(s, res_r_modflbt, '-.', 'linewidth', 2, 'color', ColMat(2,:)); 
loglog(s, res_r_qmodflbt, '--', 'linewidth', 2, 'color', ColMat(3,:)); 
leg = legend('Full-order', 'flbt', 'qflbt', 'location', 'southeast', ...
    'orientation', 'horizontal', 'interpreter', 'latex');
set(leg, 'fontsize', 10, 'interpreter', 'latex')
xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')

% Relative errors
subplot(2,1,2)
loglog(s, err_r_modflbt./res_fom, '-.', 'linewidth', 2, 'color', ColMat(2,:));
hold on
loglog(s, err_r_qmodflbt./res_fom, '--', 'linewidth', 2, 'color', ColMat(3,:)); 
leg = legend('modflbt', 'qmodflbt', 'location', 'southeast', ...
    'orientation', 'horizontal', 'interpreter', 'latex');
set(leg, 'fontsize', 10, 'interpreter', 'latex')
xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')

print -depsc2 results/qmodflbt_mag_err


%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off
