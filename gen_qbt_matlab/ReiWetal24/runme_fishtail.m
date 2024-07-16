%% RUNME_FISHTAIL
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
% From: https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Artificial_Fishtail
load('data/Fishtail.mat')
% Rayleigh Damping: D = alpha*M + beta*K
%   alpha = 1e-4;
%   beta  = 2*1e-4;
D = 1e-4*M + 2*(1e-4)*K;

%% Frequency-limited balanced truncation from data.

% Test performance from i[1e-2, 1e4]; limited frequency band of interest
% i[1e0, 1e2].
a = 0;  b = 2;  nNodes = 400;          

% Prepare quadrature weights and nodes according to Trapezoidal rule.
[nodesLeft, weightsLeft, nodesRight, weightsRight] = trapezoidal_rule([a, b], ...
    nNodes, true);

% Transfer function data.
recomputeSamples = true;
% recomputeSamples = false;
if recomputeSamples
    fprintf(1, 'COMPUTING TRANSFER FUNCTION DATA.\n')
    fprintf(1, '---------------------------------\n')
    % First-order input and output matrices.
    Bfo               = spalloc(2*n, m, nnz(B)); % Bfo = [0; B];
    Bfo(n + 1:2*n, :) = B; 
    Cfo               = spalloc(p, 2*n, nnz(C)); % Cfo = [Cso, 0];          
    Cfo(:, 1:n)       = C; 
    % Space allocation.
    sLength = length(nNodes);
    GsLeft  = zeros(p, m, nNodes);
    GsRight = zeros(p, m, nNodes);
    % Since we need the first-order solves for verifying that the Loewner
    % matrices are built correctly, we compute the data using the
    % first-order realization (as opposed to second-order).
    for k = 1:nNodes
        % Requisite linear solves.
        tic
        fprintf(1, 'Initiating structured linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------------------------\n');
        % Transfer function data.
        GsLeft(:, :, k)  = C*((nodesLeft(k)^2.*M + nodesLeft(k)*D + K)\B);
        GsRight(:, :, k) = C*((nodesRight(k)^2.*M + nodesRight(k)*D + K)\B);
        % % Gives Vright = (nodesRight(k)*Efo - Afo)\Bfo.
        % vRight           = so_structured_solve(M, D, K, B, C, nodesRight(k), 0);
        % % Gives WLeft  = ((nodesLeft(k)*Efo - Afo)')\Cfo'.
        % wLeft            = so_structured_solve(M, D, K, B, C, nodesLeft(k), 1);
        fprintf(1, 'Structured solves finished in %.2f s.\n',toc)
        fprintf(1, '-----------------------------------------------\n');
    end
    save('data/FishtailSamples_BandLimited.mat', 'GsLeft', 'GsRight', ...
        'nodesLeft', 'nodesRight')
else
    fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
    fprintf(1, '-------------------------------------------\n')
    load('data/FishtailSamples_BandLimited.mat', 'GsLeft', 'GsRight')
end

fprintf(1, 'BUILDING LOEWNER QUADRUPLE.\n')
fprintf(1, '---------------------------\n')
timeLoewner = tic;
% Loewner quadruple.
[Lbar, Mbar, Hbar, Gbar] = loewner_factory(nodesLeft, nodesRight, weightsLeft, ...
    weightsRight, GsLeft, GsRight);
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

fprintf(1, 'COMPUTING REDUCED MODEL VIA flQuadBT (non-intrusive).\n')
fprintf(1, '-----------------------------------------------------\n')
timeflBT = tic;
% Specify order of reduction.
r = 50; 
% Reduction step. 
[Hsvs, Er_flQuadBT, Ar_flQuadBT, Br_flQuadBT, Cr_flQuadBT] = quadbt_reductor(Lbar, ...
    Mbar, Hbar, Gbar, r);
Dr_flQuadBT                                                = zeros(p, m);
fprintf(1, 'flQuadBT REDUCED MODEL COMPUTED IN %.2f s\n', toc(timeflBT))
fprintf(1, '-----------------------------------------\n')

filename = 'results/FishtailROM_flQuadBT_Nodes400_r30';
save(filename, 'Hsvs', 'Er_flQuadBT', 'Ar_flQuadBT', 'Br_flQuadBT', 'Cr_flQuadBT', ...
    'Dr_flQuadBT')


%% Frequency-limited balanced truncation (intrusive).
% Input options.
opts               = ml_morlabopts('ml_ct_twostep_mor');
opts.krylovopts    = ml_morlabopts('ml_ct_krylov');
opts.mormethodopts = ml_morlabopts('ml_ct_flbt');

opts.MORMethod      = 'flbt';
opts.StoreKrylovROM = true;

% Krylov (intermediate) options.
opts.krylovopts.TwoSidedProj     = false;
opts.krylovopts.OrderComputation = 'tolerance';
opts.krylovopts.Tolerance        = sqrt(size(M, 1)) * eps;
opts.krylovopts.OutputModel      = 'so';
opts.krylovopts.StoreProjection  = true;
opts.krylovopts.krylovVopts      = struct( ...
    'NumPts'   , 500, ...
    'RealVal'  , true, ...
    'FreqRange', [-2, 4]);
opts.krylovopts.krylovWopts      = opts.krylovopts.krylovVopts;

% Frequency-limited options.
opts.mormethodopts.FreqRange        = [1.0e+00, 1.0e+02];
opts.mormethodopts.OutputModel      = 'fo';
opts.mormethodopts.OrderComputation = 'Order';
opts.mormethodopts.Order            = 30;

% Full-order system.
sys = struct('M', M, 'E', D, 'K', K, 'Bu', B, 'Cp', C);
% MOR method.
fprintf(1, 'COMPUTING REDUCED MODEL VIA flBT (intrusive, intermediate reduction).\n')
fprintf(1, '---------------------------------------------------------------------\n')
timeInterQuadBT = tic;
[rom_flBTInter, info] = ml_ct_twostep_mor(sys, opts);
fprintf(1, 'flBT REDUCED MODEL COMPUTED IN %.2f s\n', toc(timeflBT))
fprintf(1, '---------------------------------------------------------------------\n')

% Reduced-order matrices.
Er_flBTInter = rom_flBTInter.E;
Ar_flBTInter = rom_flBTInter.A;
Br_flBTInter = rom_flBTInter.B;
Cr_flBTInter = rom_flBTInter.C;
Dr_flBTInter = zeros(p, m);

interHsvs = info.infoMORMETHOD.Hsvp;
filename  = 'results/FishtailROM_flBTInter_r50';
save(filename, 'interHsvs', 'Er_flBTInter', 'Ar_flBTInter', 'Br_flBTInter', 'Cr_flBTInter', ...
    'Dr_flBTInter')

%% Plots.
numSamples     = 500;
s              = 1i*logspace(-2, 4, numSamples);
flQuadBTResp   = zeros(numSamples, 1);          % Response of (non-intrusive) flQuadBT reduced model
flQuadBTError  = zeros(numSamples, 1);          % Error due to (non-intrusive) flQuadBT reduced model
flBTInterResp  = zeros(numSamples, 1);          % Response of (intrusive, intermediate reduction) flBT reduced model
flBTInterError = zeros(numSamples, 1);          % Error due to (intrusive, intermediate reduction) flBT reduced model

% Full-order simulation data.
% recompute = false;
recompute = true;
if recompute
    Gfo     = zeros(p, m, numSamples);
    GfoResp = zeros(numSamples, 1);
    fprintf(1, 'Sampling full-order transfer function along i[1e2, 1e8].\n')
    fprintf(1, '--------------------------------------------------------\n')
    for ii = 1:numSamples
        timeSolve = tic;
        fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, s(ii))
        Gfo(:, :, ii) = C*((s(ii)^2*M +s(ii)*D + K)\B);
        GfoResp(ii)   = norm(Gfo(:, :, ii), 2);
        fprintf(1, 'k = %d solve finished in %.2f s\n', ii, toc(timeSolve))
    end
    save('data/FishtailFullOrderSimData.mat', 'Gfo', 'GfoResp')
else
    fprintf(1, 'Loading precomputed values.\n')
    fprintf(1, '--------------------------------------------------------\n')
    load('data/FishtailFullOrderSimData.mat')
end

% Compute frequency response along imaginary axis.
for ii=1:numSamples
    fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, s(ii))
    % Evaluate transfer function magnitude
    Gr_flQuadBT        = Cr_flQuadBT*((s(ii)*Er_flQuadBT - Ar_flQuadBT)\Br_flQuadBT) ...
        + Dr_flQuadBT;
    Gr_flBTInter       = Cr_flBTInter*((s(ii)*Er_flBTInter - Ar_flBTInter)\Br_flBTInter) ...
        + Dr_flBTInter;
    flQuadBTResp(ii)   = norm(Gr_flQuadBT, 2); 
    flQuadBTError(ii)  = norm(Gfo(:, :, ii) - Gr_flQuadBT, 2)/GfoResp(ii); 
    flBTInterResp(ii)  = norm(Gr_flBTInter, 2); 
    flBTInterError(ii) = norm(Gfo(:, :, ii) - Gr_flBTInter, 2)/GfoResp(ii); 
    fprintf(1, '----------------------------------------------------------------------\n');
end


plotResponse = false;
if plotResponse
    % Plot colors
    ColMat      = zeros(6,3);
    ColMat(1,:) = [0.8500    0.3250    0.0980];
    ColMat(2,:) = [0.3010    0.7450    0.9330];
    ColMat(3,:) = [0.9290    0.6940    0.1250];
    ColMat(4,:) = [0.4660    0.6740    0.1880];
    ColMat(5,:) = [0.4940    0.1840    0.5560];
    ColMat(6,:) = [1 0.4 0.6];
    
    figure(1)
    fs = 12;
    % Magnitudes
    set(gca, 'fontsize', 10)
    subplot(2,1,1)
    loglog(imag(s), GfoResp,       '-o', 'linewidth', 2, 'color', ColMat(1,:)); hold on
    loglog(imag(s), flQuadBTResp,  '--', 'linewidth', 2, 'color', ColMat(2,:)); 
    loglog(imag(s), flBTInterResp, '-.', 'linewidth', 2, 'color', ColMat(3,:)); 
    leg = legend('Full-order', 'flQuadBT', 'flBTInter', 'location', 'southeast', 'orientation', 'horizontal', ...
        'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')
    
    % Relative errors
    subplot(2,1,2)
    loglog(imag(s), flQuadBTError./GfoResp,  '-o', 'linewidth', 2, 'color', ColMat(2,:)); hold on
    loglog(imag(s), flBTInterError./GfoResp, '-*', 'linewidth', 2, 'color', ColMat(3,:));
    leg = legend('flQuadBT', 'flBTInter', 'location', 'southeast', 'orientation', 'horizontal', ...
        'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', ...
        fs, 'interpreter', 'latex')
end

write = 1;
if write
    % Store data
    resposnseMatrix = [imag(s)', GfoResp, flQuadBTResp, flBTInterResp, flBTExactResp];
    dlmwrite('results/r50FishtailResponse.dat', resposnseMatrix, ...
        'delimiter', '\t', 'precision', 8);
    errorMatrix = [imag(s)', flQuadBTError, flBTInterError, flBTExactError];
    dlmwrite('results/r50FishtailError.dat', errorMatrix, ...
        'delimiter', '\t', 'precision', 8);
end


%% Modified Gramians.
 


%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off


%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off