%% RUNME_SOFISHTAIL
% Script file to run all experiments involving the Artificial Fishtail
% model.
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


%% Problem data.
fprintf(1, 'Loading fishtail benchmark problem.\n')
fprintf(1, '----------------------------------------------\n')

% From: https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Artificial
% Fishtail.
% Rayleigh Damping: D = alpha*M + beta*K
%   alpha = 1e-4;
%   beta  = 2e-4;
load('data/Fishtail.mat')

% Rayleigh damping coefficients.
alpha = 1e-4;
beta  = 2e-4;
D     = alpha*M + beta*K;

%% Reduced order models.
% Test performance from 1e-1 to 1e-4 [Hz].
a = 0;  b = 3;  nNodes = 200;          

% Compute nodes.
omega      = 1i*(2*pi)*(logspace(0, 3, nNodes)'); % Convert to [Hz].
nodesLeft  = omega(1:2:end);    
nodesRight = omega(2:2:end); 
% Close left and right nodes under complex conjugation.
nodesLeft  = ([nodesLeft; conj(flipud(nodesLeft))]);     
nodesRight = ([nodesRight; conj(flipud(nodesRight))]);

% Prepare quadrature weights and nodes according to Trapezoidal rule.
weightsRight = [nodesRight(2) - nodesRight(1); nodesRight(3:end) - nodesRight(2:end-1); ...
    nodesRight(end) - nodesRight(end-1)]./2;
weightsRight = sqrt(1 / (2 * pi)) * sqrt(abs(weightsRight));   
weightsLeft  = [nodesLeft(2) - nodesLeft(1); nodesLeft(3:end) - nodesLeft(2:end-1); ...
    nodesLeft(end) - nodesLeft(end-1)]./2; 
weightsLeft  = sqrt(1 / (2 * pi)) * sqrt(abs(weightsLeft)); 

% Order of reduction.
r = 5;

% Transfer function data.
recomputeSamples = false;
% recomputeSamples = false;
if recomputeSamples
    fprintf(1, 'COMPUTING TRANSFER FUNCTION DATA.\n')
    fprintf(1, '---------------------------------\n')
    % Space allocation.
    GsLeft  = zeros(p, m, nNodes);
    GsRight = zeros(p, m, nNodes);
    for k = 1:nNodes
        % Requisite linear solves.
        tic
        fprintf(1, 'Linear solve %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % Transfer function data.
        GsRight(:, :, k) = C*((nodesRight(k)^2.*M + nodesRight(k).*D + K)\B);
        GsLeft(:, :, k)  = C*((nodesLeft(k)^2.*M  + nodesLeft(k).*D  + K)\B);
        fprintf(1, 'Solves finished in %.2f s.\n',toc)
        fprintf(1, '-----------------------------\n');
    end
    save('results/Fishtail_Samples_1e0to1e3Hz.mat', 'GsLeft', 'GsRight', ...
        'nodesLeft', 'nodesRight')
else
    fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
    fprintf(1, '-------------------------------------------\n')
    load('results/Fishtail_Samples_1e0to1e3Hz.mat', 'GsLeft', 'GsRight')
end

% Non-intrusive methods.
%% 1. soQuadBT.
fprintf(1, 'BUILDING LOEWNER QUADRUPLE (soQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;

% Loewner quadruple.
[Mbar_soQuadBT, Dbar_soQuadBT, Kbar_soQuadBT, Bbar_soQuadBT, CpBar_soQuadBT] = ...
    so_loewner_factory(nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeft, ...
                       GsRight, 'Rayleigh', [alpha, beta]);
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

% checkLoewner = true;
checkLoewner = false;
if checkLoewner
    fprintf(1, 'Sanity check: Verify that the build of the Loewner matrices is correct (soQuadBT).\n')
    fprintf(1, '------------------------------------------------------------------------------------------\n')

    % Quadrature-based square root factors.
    timeFactors     = tic;
    leftObsvFactor  = zeros(n, nNodes*p);
    rightContFactor = zeros(n, nNodes*m);
    fprintf(1, 'COMPUTING APPROXIMATE SQUARE-ROOT FACTORS.\n')
    fprintf(1, '------------------------------------------\n')
    for k = 1:nNodes
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k)^2.*M ...
            + nodesRight(k).*D + K)\B));

        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k))^2.*M' ...
            + conj(nodesLeft(k)).*D' + K')\C'));
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Mbar  : Error || Mbar  - leftObsvFactor.H * M * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactor' * M * rightContFactor - Mbar_soQuadBT, 2))
    fprintf('Check for Dbar  : Error || Dbar  - leftObsvFactor.H * D * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactor' * D * rightContFactor - Dbar_soQuadBT, 2))
    fprintf('Check for Kbar  : Error || Kbar  - leftObsvFactor.H * K * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactor' * K * rightContFactor - Kbar_soQuadBT, 2))
    fprintf('Check for Bbar  : Error || Bbar  - leftObsvFactor.H * B                   ||_2: %.16f\n', ...
        norm(leftObsvFactor' * B - Bbar_soQuadBT, 2))
    fprintf('Check for CpBar : Error || CpBar - Cp * rightContFactor                    ||_2: %.16f\n', ...
        norm(C * rightContFactor - CpBar_soQuadBT, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

recomputeModel = true;
if recomputeModel
    % Reductor.
    [Z_soQuadBT, S_soQuadBT, Y_soQuadBT] = svd(Mbar_soQuadBT);
    % Mr_soQuadBT  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Mbar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Mr_soQuadBT  = eye(r, r);
    Kr_soQuadBT  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Kbar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Dr_soQuadBT  = alpha*Mr_soQuadBT + beta*Kr_soQuadBT;
    Cpr_soQuadBT = CpBar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Br_soQuadBT  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Bbar_soQuadBT;
    
    filename = 'results/roFishtail_soQuadBT_r5_N200.mat';
    save(filename, 'Mr_soQuadBT', 'Dr_soQuadBT', 'Kr_soQuadBT', 'Br_soQuadBT', 'Cpr_soQuadBT');
else
    load('results/roFishtail_soQuadBT_r5_N200.mat')
end

%% 2. soLoewner.
fprintf(1, 'BUILDING LOEWNER QUADRUPLE (soLoewner).\n')
fprintf(1, '---------------------------------------\n')
timeLoewner = tic;
[Mbar_soLoewner, Dbar_soLoewner, Kbar_soLoewner, Bbar_soLoewner, CpBar_soLoewner] = ...
    so_loewner_factory(nodesLeft, nodesRight, ones(nNodes, 1), ones(nNodes, 1), GsLeft, ...
                       GsRight, 'Rayleigh', [alpha, beta]);
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

% checkLoewner = true;
checkLoewner = false;
if checkLoewner
    fprintf(1, 'Sanity check: Verify that the build of the Loewner matrices is correct (soLoewner).\n')
    fprintf(1, '------------------------------------------------------------------------------------------\n')

    % Quadrature-based square root factors.
    timeFactors     = tic;
    leftObsvFactor  = zeros(n, nNodes*p);
    rightContFactor = zeros(n, nNodes*m);
    fprintf(1, 'COMPUTING APPROXIMATE SQUARE-ROOT FACTORS.\n')
    fprintf(1, '------------------------------------------\n')
    for k = 1:nNodes
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = (((nodesRight(k)^2.*M ...
            + nodesRight(k).*D + K)\B));

        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = (((conj(nodesLeft(k))^2.*M' ...
            + conj(nodesLeft(k)).*D' + K')\C'));
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Mbar  : Error || Mbar  - leftObsvFactor.H * M * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactor' * M * rightContFactor - Mbar_soLoewner, 2))
    fprintf('Check for Dbar  : Error || Dbar  - leftObsvFactor.H * D * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactor' * D * rightContFactor - Dbar_soLoewner, 2))
    fprintf('Check for Kbar  : Error || Kbar  - leftObsvFactor.H * K * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactor' * K * rightContFactor - Kbar_soLoewner, 2))
    fprintf('Check for Bbar  : Error || Bbar  - leftObsvFactor.H * B                   ||_2: %.16f\n', ...
        norm(leftObsvFactor' * B - Bbar_soLoewner, 2))
    fprintf('Check for CpBar : Error || CpBar - Cp * rightContFactor                   ||_2: %.16f\n', ...
        norm(C * rightContFactor - CpBar_soLoewner, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end


recomputeModel = true;
if recomputeModel
    % Reductor.
    % Relevant SVDs.
    [Yl_soLoewner, Sl_soLoewner, ~] = svd([-Mbar_soLoewner, Kbar_soLoewner], 'econ');
    [~, Sr_soLoewner, Xr_soLoewner] = svd([-Mbar_soLoewner; Kbar_soLoewner], 'econ');
    
    % Compress.
    Mr_soLoewner  = Yl_soLoewner(:, 1:r)'*Mbar_soLoewner*Xr_soLoewner(:, 1:r); % This needs a -?
    Kr_soLoewner  = Yl_soLoewner(:, 1:r)'*Kbar_soLoewner*Xr_soLoewner(:, 1:r);
    Dr_soLoewner  = alpha*Mr_soLoewner + beta*Kr_soLoewner;
    Br_soLoewner  = Yl_soLoewner(:, 1:r)'*Bbar_soLoewner;
    Cpr_soLoewner = CpBar_soLoewner*Xr_soLoewner(:, 1:r);
    
    filename = 'results/roFishtail_soLoewner_r5_N200.mat';
    save(filename, 'Mr_soLoewner', 'Dr_soLoewner', 'Kr_soLoewner', 'Br_soLoewner', 'Cpr_soLoewner');
else
    load('results/roFishtail_soLoewner_r5_N200.mat')
end

%% 3. foQuadBT.
fprintf(1, 'BUILDING LOEWNER QUADRUPLE (foQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;
% Loewner quadruple.
[Ebar_foQuadBT, Abar_foQuadBT, Bbar_foQuadBT, Cbar_foQuadBT] = fo_loewner_factory(...
    nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeft, GsRight);
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

% checkLoewner = true;
checkLoewner = false;
if checkLoewner
    fprintf(1, 'Sanity check: Verify that the build of the Loewner matrices is correct (foQuadBT).\n')
    fprintf(1, '------------------------------------------------------------------------------------------\n')

    % Quadrature-based square root factors.
    timeFactors     = tic;
    leftObsvFactor  = zeros(2*n, nNodes*p);
    rightContFactor = zeros(2*n, nNodes*m);
    fprintf(1, 'COMPUTING APPROXIMATE SQUARE-ROOT FACTORS.\n')
    fprintf(1, '------------------------------------------\n')
    
    % Lift realization for comparison.
    Efo                   = spalloc(2*n, 2*n, nnz(M) + n); % Descriptor matrix; Efo = [I, 0: 0, M]
    Efo(1:n, 1:n)         = speye(n);                      % (1, 1) block
    Efo(n+1:2*n, n+1:2*n) = M;                             % (2, 2) block is (sparse) mass matrix M
    
    Afo                   = spalloc(2*n, 2*n, nnz(K) + nnz(D) + n); % Afo = [0, I; -K, -D]
    Afo(1:n, n+1:2*n)     = speye(n);                               % (1, 2) block of Afo
    Afo(n+1:2*n, 1:n)     = -K;                                     % (2, 1) block is -K
    Afo(n+1:2*n, n+1:2*n) = -D;                                     % (2, 2) block is -D 
    
    Bfo             = spalloc(2*n, 1, nnz(B)); % Bfo = [0; B];
    Bfo(n+1:2*n, :) = B; 

    Cfo         = spalloc(12, 2*n, nnz(C)); % Bfo = [Cp, 0];
    Cfo(:, 1:n) = C; 


    for k = 1:nNodes
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k).*Efo - Afo)\Bfo));

        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k)).*Efo' - Afo')\Cfo'));
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Ebar : Error || Ebar  - leftObsvFactor.H * Efo * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactor' * Efo * rightContFactor - Ebar_foQuadBT, 2))
    fprintf('Check for Abar : Error || Abar  - leftObsvFactor.H * Afo * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactor' * Afo * rightContFactor - Abar_foQuadBT, 2))
    fprintf('Check for Bbar : Error || Bbar  - leftObsvFactor.H * Bfo                   ||_2: %.16f\n', ...
        norm(leftObsvFactor' * Bfo - Bbar_foQuadBT, 2))
    fprintf('Check for Cbar : Error || Cbar - C * rightContFactor                       ||_2: %.16f\n', ...
        norm(Cfo * rightContFactor - Cbar_foQuadBT, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

recomputeModel = true;
if recomputeModel
    % Reductor.
    [Z_foQuadBT, S_foQuadBT, Y_foQuadBT] = svd(Ebar_foQuadBT);
    Er_foQuadBT  = eye(r, r);
    Ar_foQuadBT  = (S_foQuadBT(1:r, 1:r)^(-1/2)*Z_foQuadBT(:, 1:r)')*Abar_foQuadBT*(Y_foQuadBT(:, 1:r)*S_foQuadBT(1:r, 1:r)^(-1/2));
    Cr_foQuadBT  = Cbar_foQuadBT*(Y_foQuadBT(:, 1:r)*S_foQuadBT(1:r, 1:r)^(-1/2));
    Br_foQuadBT  = (S_foQuadBT(1:r, 1:r)^(-1/2)*Z_foQuadBT(:, 1:r)')*Bbar_foQuadBT;
    
    filename = 'results/roFishtail_foQuadBT_r5_N200.mat';
    save(filename, 'Er_foQuadBT', 'Ar_foQuadBT', 'Br_foQuadBT', 'Cr_foQuadBT');
else
    load('results/roFishtail_foQuadBT_r5_N200.mat')
end

%% 4. soBT.
recomputeModel = false;
if recomputeModel
    soSys    = struct();
    soSys.M  = M;
    soSys.E  = D;
    soSys.K  = K;
    soSys.Bu = B;
    soSys.Cp = C;
    soSys.Cv = zeros(p, n);
    soSys.D  = zeros(p, m);
    
    % Input opts.
    opts                  = struct();
    opts.BalanceType      = 'pv';
    opts.Order            = r;
    opts.OrderComputation = 'order';
    opts.OutputModel      = 'so';
    
    [soBTRom_Rayleigh, info] = ml_ct_s_soss_bt(soSys, opts);
    Mr_soBT = soBTRom_Rayleigh.M;
    Dr_soBT = soBTRom_Rayleigh.E;
    Kr_soBT = soBTRom_Rayleigh.K;
    Br_soBT = soBTRom_Rayleigh.Bu;
    Cpr_soBT = soBTRom_Rayleigh.Cp;
    
    filename = 'results/roFishtail_soBT_r5.mat';
    save(filename, 'Mr_soBT', 'Dr_soBT', 'Kr_soBT', 'Br_soBT', 'Cpr_soBT');
else
    load('results/roFishtail_soBT_r5.mat')
end

%% Plots.
numSamples      = 500;
s               = 1i*(2*pi)*logspace(-4, 4, numSamples);
s_hz            = imag(s)/2/pi;

% Transfer function evaluations.
Gr_soQuadBT  = zeros(p, m, numSamples);
Gr_soLoewner = zeros(p, m, numSamples);
Gr_foQuadBT  = zeros(p, m, numSamples);
Gr_soBT      = zeros(p, m, numSamples);

% Magnitude response and errors.
resp_soQuadBT          = zeros(numSamples, 1); % Response of (non-intrusive) soQuadBT reduced model
relSVError_soQuadBT    = zeros(numSamples, 1); % Rel. SV error due to (non-intrusive) soQuadBT reduced model
absSVError_soQuadBT    = zeros(numSamples, 1); % Abs. SV error due to (non-intrusive) soQuadBT reduced model
absFrobError_soQuadBT  = zeros(numSamples, 1); % Abs. Frob. error due to (non-intrusive) soQuadBT reduced model

resp_soLoewner         = zeros(numSamples, 1); % Response of (non-intrusive) soLoewner reduced model
relSVError_soLoewner   = zeros(numSamples, 1); % Rel. SV error due to (non-intrusive) soLoewner reduced model
absSVError_soLoewner   = zeros(numSamples, 1); % Abs. SV error due to (non-intrusive) soLoewner reduced model
absFrobError_soLoewner = zeros(numSamples, 1); % Abs. Frob. error due to (non-intrusive) soQuadBT reduced model

resp_foQuadBT          = zeros(numSamples, 1); % Response of (non-intrusive) foQuadBT reduced model
relSVError_foQuadBT    = zeros(numSamples, 1); % Rel. SV error due to (non-intrusive) foQuadBT reduced model
absSVError_foQuadBT    = zeros(numSamples, 1); % Abs. SV error due to (non-intrusive) foQuadBT reduced model
absFrobError_foQuadBT  = zeros(numSamples, 1); % Abs. Frob. error due to (non-intrusive) soQuadBT reduced model

resp_soBT              = zeros(numSamples, 1); % Response of (intrusive, intermediate reduction) soBT reduced model
relSVError_soBT        = zeros(numSamples, 1); % Rel. SV error due to (intrusive, intermediate reduction) soBT reduced model
absSVError_soBT        = zeros(numSamples, 1); % Abs. SV error due to (intrusive, intermediate reduction) soBT reduced model
absFrobError_soBT      = zeros(numSamples, 1); % Abs. Frob. error due to (non-intrusive) soQuadBT reduced model


% Full-order simulation data.
recompute = true;
% recompute = false;
if recompute
    Gfo     = zeros(p, m, numSamples);
    GfoResp = zeros(numSamples, 1);
    GfoFrob = zeros(numSamples, 1);
    fprintf(1, 'Sampling full-order transfer function along i[1e2, 1e7].\n')
    fprintf(1, '--------------------------------------------------------\n')
    for ii = 1:numSamples
        timeSolve = tic;
        fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, s(ii))
        Gfo(:, :, ii) = C*((s(ii)^2*M +s(ii)*D + K)\B);
        GfoResp(ii)   = max(svd(Gfo(:, :, ii)));
        GfoFrob(ii)   = norm(Gfo(:, :, ii), 'fro');     % Matrix Frobenius-norm
        fprintf(1, 'k = %d solve finished in %.2f s\n', ii, toc(timeSolve))
    end
    save('results/FishtailFullOrderSimData_1e-4to1e4Hz.mat', 'Gfo', 'GfoResp', 'GfoFrob')
else
    fprintf(1, 'Loading precomputed values.\n')
    fprintf(1, '--------------------------------------------------------\n')
    load('results/FishtailFullOrderSimData_1e-4to1e4Hz.mat')
end

% Compute frequency response along imaginary axis.
for ii=1:numSamples
    fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, real(s(ii)))
    % Transfer functions.
    Gr_soQuadBT(:, :, ii)    = Cpr_soQuadBT*((s(ii)^2*Mr_soQuadBT + s(ii)*Dr_soQuadBT + Kr_soQuadBT)\Br_soQuadBT);
    Gr_foQuadBT(:, :, ii)    = Cr_foQuadBT*((s(ii)*Er_foQuadBT + Ar_foQuadBT)\Br_foQuadBT);
    Gr_soLoewner(:, :, ii)   = Cpr_soLoewner*((s(ii)^2*Mr_soLoewner + s(ii)*Dr_soLoewner + Kr_soLoewner)\Br_soLoewner);
    Gr_soBT(:, :, ii)        = Cpr_soBT*((s(ii)^2*Mr_soBT + s(ii)*Dr_soBT + Kr_soBT)\Br_soBT);

    % Response and errors. 
    resp_soQuadBT(ii)          = norm(Gr_soQuadBT(:, :, ii), 2); 
    absSVError_soQuadBT(ii)    = max(svd(Gfo(:, :, ii) - Gr_soQuadBT(:, :, ii)));
    relSVError_soQuadBT(ii)    = absSVError_soQuadBT(ii)/GfoResp(ii);
    absFrobError_soQuadBT(ii)  = norm((Gfo(:, :, ii) - Gr_soQuadBT(:, :, ii)), 'fro');

    resp_foQuadBT(ii)          = norm(Gr_foQuadBT(:, :, ii), 2); 
    absSVError_foQuadBT(ii)    = max(svd(Gfo(:, :, ii) - Gr_foQuadBT(:, :, ii))); 
    relSVError_foQuadBT(ii)    = absSVError_foQuadBT(ii)/GfoResp(ii); 
    absFrobError_foQuadBT(ii)  = norm((Gfo(:, :, ii) - Gr_foQuadBT(:, :, ii)), 'fro');

    resp_soLoewner(ii)         = norm(Gr_soLoewner(:, :, ii), 2); 
    absSVError_soLoewner(ii )  = max(svd(Gfo(:, :, ii) - Gr_soLoewner(:, :, ii)));
    relSVError_soLoewner(ii)   = absSVError_soLoewner(ii)/GfoResp(ii); 
    absFrobError_soLoewner(ii) = norm((Gfo(:, :, ii) - Gr_soLoewner(:, :, ii)), 'fro');

    resp_soBT(ii)              = norm(Gr_soBT(:, :, ii), 2); 
    absSVError_soBT(ii)        = max(svd(Gfo(:, :, ii) - Gr_soBT(:, :, ii)));
    relSVError_soBT(ii)        = absSVError_soBT(ii)/GfoResp(ii); 
    absFrobError_soBT(ii)      = norm((Gfo(:, :, ii) - Gr_soBT(:, :, ii)), 'fro');

    fprintf(1, '----------------------------------------------------------------------\n');
end

plotResponse = true;
% plotResponse = false;
if plotResponse
    % Plot colors
    ColMat      = zeros(6,3);
    ColMat(1,:) = [0.8500    0.3250    0.0980];
    ColMat(2,:) = [0.3010    0.7450    0.9330];
    ColMat(3,:) = [0.9290    0.6940    0.1250];
    ColMat(4,:) = [0.4660    0.6740    0.1880];
    ColMat(5,:) = [0.4940    0.1840    0.5560];
    ColMat(6,:) = [1         0.4       0.6];
    
    figure(1)
    fs = 12;
    % Magnitudes
    set(gca, 'fontsize', 10)
    subplot(2,1,1)
    loglog(s_hz, GfoResp,       '-o', 'linewidth', 2, 'color', ColMat(1,:)); hold on
    loglog(s_hz, resp_soQuadBT,  '--', 'linewidth', 2, 'color', ColMat(2,:)); 
    loglog(s_hz, resp_foQuadBT,  '-.', 'linewidth', 2, 'color', ColMat(3,:)); 
    loglog(s_hz, resp_soLoewner,  '--', 'linewidth', 2, 'color', ColMat(4,:)); 
    loglog(s_hz, resp_soBT, '-.', 'linewidth', 2, 'color', ColMat(5,:)); 
    leg = legend('Full-order', 'soQuadBT', 'foQuadBT', 'soLoewner', 'soBT', 'location', 'southeast', 'orientation', 'horizontal', ...
        'interpreter', 'latex');
    xlim([(s_hz(1)), (s_hz(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('s [Hz]', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')
    
    % Relative errors
    subplot(2,1,2)
    loglog(s_hz, relSVError_soQuadBT,  '-o', 'linewidth', 2, 'color', ColMat(2,:)); hold on
    loglog(s_hz, relSVError_foQuadBT,  '-*', 'linewidth', 2, 'color', ColMat(3,:));
    loglog(s_hz, relSVError_soLoewner, '-*', 'linewidth', 2, 'color', ColMat(4,:));
    loglog(s_hz, relSVError_soBT,      '-*', 'linewidth', 2, 'color', ColMat(5,:));
    leg = legend('soQuadBT', 'foQuadBT', 'soLoewner', 'soBT', 'location', 'southeast', 'orientation', 'horizontal', ...
        'interpreter', 'latex');
    xlim([(s_hz(1)), (s_hz(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('s [Hz]', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', ...
        fs, 'interpreter', 'latex')
end

% Store data.
write = true;
if write
    magMatrix = [s_hz', GfoResp, resp_soQuadBT, resp_foQuadBT, resp_soLoewner, ...
        resp_soBT];
    dlmwrite('results/FishtailReducedOrderResponse_r5_N200.dat', magMatrix, ...
        'delimiter', '\t', 'precision', 8);
    errorMatrix = [s_hz', relSVError_soQuadBT, relSVError_foQuadBT, relSVError_soLoewner, relSVError_soBT];
    dlmwrite('results/FishtailReducedOrderError_r5_N200.dat', errorMatrix, ...
        'delimiter', '\t', 'precision', 8);
end


%% Error measures.
% Print errors.
fprintf(1, 'Order r = %d.\n', r)
fprintf(1, '--------------\n')
fprintf(1, 'Relative H-infty error due to soQuadBT : %.16f \n', max((absSVError_soQuadBT))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to soLoewner: %.16f \n', max((absSVError_soLoewner))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to foLoewner: %.16f \n', max((absSVError_foQuadBT))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to soBT     : %.16f \n', max((absSVError_soBT))./max((GfoResp)))
fprintf(1, '------------------------------------------------------------\n')
fprintf(1, 'Relative H-2 error due to soQuadBT : %.16f \n', sum(absFrobError_soQuadBT)/sum(GfoFrob))
fprintf(1, 'Relative H-2 error due to soLoewner: %.16f \n', sum(absFrobError_soLoewner)/sum(GfoFrob))
fprintf(1, 'Relative H-2 error due to foLoewner: %.16f \n', sum(absFrobError_foQuadBT)/sum(GfoFrob))
fprintf(1, 'Relative H-2 error due to soBT     : %.16f \n', sum(absFrobError_soBT)/sum(GfoFrob))
fprintf(1, '------------------------------------------------------------\n')


%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off
