%% RUNME_SOMSD
% Script file to run all experiments involving toy mass-spring-damper
% system.
%

%
% This file is part of the archive Code, Data, and Results for Numerical 
% Experiments in "..."
% Copyright (c) 2024 Sean Reiter,
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


%% Load toy problem.
fprintf(1, 'Toy mass-spring-damper system.\n')
fprintf(1, '------------------------------\n');

write = 1;
if write
    % 1. Rayleigh damping: D = alpha*M + beta*K
    n1 = 100;   alpha = .002;   beta = alpha;   v = 0;
    [M, DRayleigh, K] = triplechain_MSD(n1, alpha, beta, v);
    
    % Input, output, state dimensions.
    n = size(full(K), 1);   p = 1;  m = 1;
    
    % Input and output matrices.
    B  = ones(n, m);   
    C  = ones(p, n);
    save('data/MSDRayleigh.mat', 'M', 'DRayleigh', 'K', 'C', 'B', 'n', 'm', 'p')

    % 2. Structural damping: D(s) = (1i/s)*eta*K
    eta     = 1e-2;
    DStruct = 1i*eta*K;
    save('data/MSDStruct.mat', 'M', 'DStruct', 'K', 'C', 'B', 'n', 'm', 'p')
else
    load('data/MSDRayleigh.mat')
    load('data/MSDStruct.mat')
end


%% Inverse tests.
% First-order companion form.

% Descriptor matrix.
Efo                       = zeros(2*n, 2*n); % Descriptor matrix; Efo = [I, 0: 0, M]
Efo(1:n, 1:n)             = eye(n);          % (1, 1) block
Efo(n + 1:2*n, n + 1:2*n) = M;               % (2, 2) block is mass matrix

% State matrix.
Afo                       = zeros(2*n, 2*n); % Afo = [0, I; -Kso, -Dso]
Afo(1:n, n + 1:2*n)       = eye(n);          % (1, 2) block of Afo
Afo(n + 1:2*n, 1:n)       = -K;              % (2, 1) block is -K
Afo(n + 1:2*n, n + 1:2*n) = -DRayleigh;      % (2, 2) block is -D

% Input matrix.
Bfo               = zeros(2*n, m); % Bfo = [0; B];
Bfo(n + 1:2*n, :) = B; 

% Output matrix. Position input, only
Cfo         = zeros(p, 2*n); % Cfo = [C, 0];
Cfo(:, 1:n) = C; 

% Test frequency. 
s = 1i*1e-1;

% 1. Inverse using MATLAB's \.
resolventInverseRef = (s*Efo-Afo)\eye(2*n, 2*n);

% 2. Using formula (a) (assumes s*M + D is invertible).
resolventInverseA = zeros(2*n, 2*n);
tmpInv1           = (s^2*M + s*DRayleigh + K)\eye(n, n);
tmpInv2           = (s*M + DRayleigh)\eye(n, n);

resolventInverseA(1:n, 1:n)             = tmpInv1*(s*M + DRayleigh);
resolventInverseA(1:n, n+1:2*n)         = tmpInv1;
resolventInverseA(n + 1:2*n, 1:n)       = -tmpInv2*K*tmpInv1*(s*M + DRayleigh);
resolventInverseA(n + 1:2*n, n + 1:2*n) = tmpInv2*(eye(n, n) - K*tmpInv1);

% 3. Using formula (b) (assumes s*I is invertible).
resolventInverseB = zeros(2*n, 2*n);

resolventInverseB(1:n, 1:n)             = (1/s)*(eye(n, n) - tmpInv1*K);
resolventInverseB(1:n, n+1:2*n)         = tmpInv1;
resolventInverseB(n + 1:2*n, 1:n)       = -tmpInv1*K;
resolventInverseB(n + 1:2*n, n + 1:2*n) = s*tmpInv1;

fprintf(1, 'RELATIVE ERROR IN FORMULA (a): %.16f\n', norm(resolventInverseRef - resolventInverseA, 2)/norm(resolventInverseRef, 2))
fprintf(1, '------------------------------------------------\n')

fprintf(1, 'RELATIVE ERROR IN FORMULA (b): %.16f\n', norm(resolventInverseRef - resolventInverseB, 2)/norm(resolventInverseRef, 2))
fprintf(1, '------------------------------------------------\n')

%% Prepare quadrature weights and nodes.
% Test performance from i[1e0, 1e4].
a = -3;  b = 4;  nNodes = 100;          

% Prepare quadrature weights and nodes according to Trapezoidal rule.
[nodesLeft, weightsLeft, nodesRight, weightsRight] = trapezoidal_rule([a, b], ...
    nNodes, true);


%% 1. Rayleigh Damping.

% Transfer function data.
recomputeSamples = true;
% recomputeSamples = false;
if recomputeSamples
    fprintf(1, 'COMPUTING TRANSFER FUNCTION DATA.\n')
    fprintf(1, '---------------------------------\n')
    % Space allocation.
    sLength = length(nNodes);
    GsLeftRayleigh  = zeros(p, m, nNodes);
    GsRightRayleigh = zeros(p, m, nNodes);
    for k = 1:nNodes
        % Requisite linear solves.
        % tic
        % fprintf(1, 'Linear solve %d of %d.\n', k, nNodes)
        % fprintf(1, '-----------------------------\n');
        % Transfer function data.
        GsRightRayleigh(:, :, k) = C*((nodesRight(k)^2.*M + nodesRight(k).*DRayleigh + K)\B);
        GsLeftRayleigh(:, :, k)  = C*((nodesLeft(k)^2.*M  + nodesLeft(k).*DRayleigh  + K)\B);
        % fprintf(1, 'Solves finished in %.2f s.\n',toc)
        % fprintf(1, '-----------------------------\n');
    end
    save('data/MSDRayleigh_Samples_1e0to1e4.mat', 'GsLeftRayleigh', 'GsRightRayleigh', ...
        'nodesLeft', 'nodesRight')
else
    fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
    fprintf(1, '-------------------------------------------\n')
    load('data/MSDRayleigh_Samples_1e0to1e4.mat', 'GsLeftRayleigh', 'GsRightRayleigh')
end

fprintf(1, 'BUILDING LOEWNER QUADRUPLE.\n')
fprintf(1, '---------------------------\n')
timeLoewner = tic;
% Loewner quadruple.
[Lbar_M_Rayleigh, Lbar_D_Rayleigh, Lbar_K_Rayleigh, Hbar_Rayeigh, Gbar_Rayleigh] = ...
    so_loewner_factory(nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeftRayleigh, ...
                       GsRightRayleigh, 'Rayleigh', [alpha, beta]);
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

checkLoewner = true;
% checkLoewner = false;
if checkLoewner
    fprintf(1, 'Sanity check: Verify that the build of the Loewner matrices is correct (Rayleigh Damping).\n')
    fprintf(1, '------------------------------------------------------------------------------------------\n')

    % Quadrature-based square root factors.
    timeFactors     = tic;
    leftObsvFactorRayleigh  = zeros(n, nNodes*p);
    rightContFactorRayleigh = zeros(n, nNodes*m);
    fprintf(1, 'COMPUTING APPROXIMATE SQUARE-ROOT FACTORS.\n')
    fprintf(1, '------------------------------------------\n')
    for k = 1:nNodes
        % For (position) controllability Gramian.
        rightContFactorRayleigh(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k)^2.*M ...
            + nodesRight(k).*DRayleigh + K)\B));

        % For (velocity) observability Gramian.
        leftObsvFactorRayleigh(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k))^2.*M' ...
            + conj(nodesLeft(k)).*DRayleigh' + K')\C'));
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Lbar_M_Rayleigh: Error || Lbar_M_Rayleigh - leftObsvFactor.H * M * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactorRayleigh' * M * rightContFactorRayleigh - Lbar_M_Rayleigh, 2))
    fprintf('Check for Lbar_D_Rayleigh: Error || Lbar_D_Rayleigh - leftObsvFactor.H * D * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactorRayleigh' * DRayleigh * rightContFactorRayleigh - Lbar_D_Rayleigh, 2))
    fprintf('Check for Lbar_K_Rayleigh: Error || Lbar_K_Rayleigh - leftObsvFactor.H * K * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactorRayleigh' * K * rightContFactorRayleigh - Lbar_K_Rayleigh, 2))
    fprintf('Check for Hbar_Rayleigh  : Error || Hbar_Rayleigh   - leftObsvFactor.H * B                   ||_2: %.16f\n', ...
        norm(leftObsvFactorRayleigh' * B - Hbar_Rayeigh, 2))
    fprintf('Check for Gbar_Rayleigh  : Error || Gbar_Rayleigh   - C * rightContFactor                    ||_2: %.16f\n', ...
        norm(C * rightContFactorRayleigh - Gbar_Rayleigh, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

r = 20;

% Intrusive. 
soSys_Rayleigh    = struct();
soSys_Rayleigh.M  = M;
soSys_Rayleigh.E  = DRayleigh;
soSys_Rayleigh.K  = K;
soSys_Rayleigh.Bu = B;
soSys_Rayleigh.Cp = C;
soSys_Rayleigh.Cv = zeros(p, n);
soSys_Rayleigh.D  = zeros(p, m);

% Input opts.
opts                  = struct();
opts.BalanceType      = 'pv';
opts.Order            = r;
opts.OrderComputation = 'order';
opts.OutputModel      = 'so';

[soBTRom_Rayleigh, info] = ml_ct_d_soss_bt(soSys_Rayleigh, opts);
MrBT_Rayleigh = soBTRom_Rayleigh.M;
DrBT_Rayleigh = soBTRom_Rayleigh.E;
KrBT_Rayleigh = soBTRom_Rayleigh.K;
BrBT_Rayleigh = soBTRom_Rayleigh.Bu;
CrBT_Rayleigh = soBTRom_Rayleigh.Cp;

% Non-intrusive. (Loewner matrices computed above.)
[Z, S, Y] = svd(Lbar_M_Rayleigh);
MrQuadBT_Rayleigh = (S(1:r, 1:r)^(-1/2)*Z(:, 1:r)')*Lbar_M_Rayleigh*(Y(:, 1:r)*S(1:r, 1:r)^(-1/2));
DrQuadBT_Rayleigh = (S(1:r, 1:r)^(-1/2)*Z(:, 1:r)')*Lbar_D_Rayleigh*(Y(:, 1:r)*S(1:r, 1:r)^(-1/2));
KrQuadBT_Rayleigh = (S(1:r, 1:r)^(-1/2)*Z(:, 1:r)')*Lbar_K_Rayleigh*(Y(:, 1:r)*S(1:r, 1:r)^(-1/2));
CrQuadBT_Rayleigh = Gbar_Rayleigh*(Y(:, 1:r)*S(1:r, 1:r)^(-1/2));
BrQuadBT_Rayleigh = (S(1:r, 1:r)^(-1/2)*Z(:, 1:r)')*Hbar_Rayeigh;

% Plots.
numSamples     = 500;
s              = 1i*logspace(-3, 4, numSamples);
soQuadBTResp   = zeros(numSamples, 1);          % Response of (non-intrusive) soQuadBT reduced model
soQuadBTError  = zeros(numSamples, 1);          % Error due to (non-intrusive) flQuadBT reduced model
soBTResp       = zeros(numSamples, 1);          % Response of (intrusive, intermediate reduction) soBT reduced model
soBTError      = zeros(numSamples, 1);          % Error due to (intrusive, intermediate reduction) soBT reduced model

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
        % fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, s(ii))
        Gfo(:, :, ii) = C*((s(ii)^2*M +s(ii)*DRayleigh + K)\B);
        GfoResp(ii)   = norm(Gfo(:, :, ii), 2);
        % fprintf(1, 'k = %d solve finished in %.2f s\n', ii, toc(timeSolve))
    end
    % save('data/ButterflyFullOrderSimData.mat', 'Gfo', 'GfoResp')
else
    % fprintf(1, 'Loading precomputed values.\n')
    % fprintf(1, '--------------------------------------------------------\n')
    % load('data/ButterflyFullOrderSimData.mat')
end

% Compute frequency response along imaginary axis.
for ii=1:numSamples
    fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, s(ii))
    % Evaluate transfer function magnitude
    Gr_soQuadBT        = CrQuadBT_Rayleigh*((s(ii)^2*MrQuadBT_Rayleigh + s(ii)*DrQuadBT_Rayleigh + KrQuadBT_Rayleigh)\BrQuadBT_Rayleigh);
    Gr_soBT            = CrBT_Rayleigh*((s(ii)^2*MrBT_Rayleigh + s(ii)*DrBT_Rayleigh + KrBT_Rayleigh)\BrBT_Rayleigh);
    soQuadBTResp(ii)   = norm(Gr_soQuadBT, 2); 
    soQuadBTError(ii)  = norm(Gfo(:, :, ii) - Gr_soQuadBT, 2)/GfoResp(ii); 
    soBTResp(ii)  = norm(Gr_soBT, 2); 
    soBTError(ii) = norm(Gfo(:, :, ii) - Gr_soBT, 2)/GfoResp(ii); 
    fprintf(1, '----------------------------------------------------------------------\n');
end


plotResponse = true;
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
    loglog(imag(s), soQuadBTResp,  '--', 'linewidth', 2, 'color', ColMat(2,:)); 
    loglog(imag(s), soBTResp, '-.', 'linewidth', 2, 'color', ColMat(3,:)); 
    leg = legend('Full-order', 'soQuadBT', 'soBT', 'location', 'southeast', 'orientation', 'horizontal', ...
        'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')
    
    % Relative errors
    subplot(2,1,2)
    loglog(imag(s), soQuadBTError./GfoResp,  '-o', 'linewidth', 2, 'color', ColMat(2,:)); hold on
    loglog(imag(s), soBTError./GfoResp, '-*', 'linewidth', 2, 'color', ColMat(3,:));
    leg = legend('soQuadBT', 'soBT', 'location', 'southeast', 'orientation', 'horizontal', ...
        'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', ...
        fs, 'interpreter', 'latex')
end

%% 2. Structural Damping.

% Transfer function data.
recomputeSamples = true;
% recomputeSamples = false;
if recomputeSamples
    fprintf(1, 'COMPUTING TRANSFER FUNCTION DATA.\n')
    fprintf(1, '---------------------------------\n')
    % Space allocation.
    sLength = length(nNodes);
    GsLeftStruct  = zeros(p, m, nNodes);
    GsRightStruct = zeros(p, m, nNodes);
    for k = 1:nNodes
        % Requisite linear solves.
        % tic
        % fprintf(1, 'Linear solve %d of %d.\n', k, nNodes)
        % fprintf(1, '-----------------------------\n');
        % Transfer function data.
        % (1/s) factor in DStruct(s) and s cancel out in evaluation.
        GsRightStruct(:, :, k) = C*((nodesRight(k)^2.*M + DStruct + K)\B);
        GsLeftStruct(:, :, k)  = C*((nodesLeft(k)^2.*M  + DStruct + K)\B);
        % fprintf(1, 'Solves finished in %.2f s.\n',toc)
        % fprintf(1, '-----------------------------\n');
    end
    save('data/MSDStructural_Samples_1e0to1e4.mat', 'GsLeftStruct', 'GsRightStruct', ...
        'nodesLeft', 'nodesRight')
else
    fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
    fprintf(1, '-------------------------------------------\n')
    load('data/MSDStructural_Samples_1e0to1e4.mat', 'GsLeftStruct', 'GsRightStruct')
end

fprintf(1, 'BUILDING LOEWNER QUADRUPLE.\n')
fprintf(1, '---------------------------\n')
timeLoewner = tic;
% Loewner quadruple.
[Lbar_M_Struct, Lbar_D_Struct, Lbar_K_Struct, Hbar_Struct, Gbar_Struct] = ...
    so_loewner_factory(nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeftStruct, ...
                       GsRightStruct, 'Structural', eta);
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

checkLoewner = true;
% checkLoewner = false;
if checkLoewner
    fprintf(1, 'Sanity check: Verify that the build of the Loewner matrices is correct (Structural damping).\n')
    fprintf(1, '--------------------------------------------------------------------------------------------\n')

    % Quadrature-based square root factors.
    timeFactors     = tic;
    leftObsvFactorStruct  = zeros(n, nNodes*p);
    rightContFactorStruct = zeros(n, nNodes*m);
    fprintf(1, 'COMPUTING APPROXIMATE SQUARE-ROOT FACTORS.\n')
    fprintf(1, '------------------------------------------\n')
    for k = 1:nNodes
        % For (position) controllability Gramian.
        rightContFactorStruct(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k)^2.*M ...
            + DStruct + K)\B));

        % For (velocity) observability Gramian.
        leftObsvFactorStruct(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k))^2.*M' ...
            + DStruct' + K')\C'));
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Lbar_M_Struct: Error || Lbar_M_Struct - leftObsvFactor.H * M * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactorStruct' * M * rightContFactorStruct - Lbar_M_Struct, 2))
    fprintf('Check for Lbar_D_Struct: Error || Lbar_D_Struct - leftObsvFactor.H * D * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactorStruct' * DStruct * rightContFactorStruct - Lbar_D_Struct, 2))
    fprintf('Check for Lbar_K_Struct: Error || Lbar_K_Struct - leftObsvFactor.H * K * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactorStruct' * K * rightContFactorStruct - Lbar_K_Struct, 2))
    fprintf('Check for Hbar_Struct  : Error || Hbar_Struct   - leftObsvFactor.H * B                   ||_2: %.16f\n', ...
        norm(leftObsvFactorStruct' * B - Hbar_Struct, 2))
    fprintf('Check for Gbar_Struct  : Error || Gbar_Struct   - C * rightContFactor                    ||_2: %.16f\n', ...
        norm(C * rightContFactorStruct - Gbar_Struct, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end


%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off
