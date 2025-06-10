%% RUNME_SOBUTTERFLY
% Script file to run all experiments involving the Butterfly Gyroscope
% model.
%

%
% This file is part of the archive Code, Data and Results for Numerical 
% Experiments in "Data-driven balanced truncation for second-order 
% systems with generalized proportional damping"
% Copyright (c) 2025 Sean Reiter, Steffen W. R. Werner
% All rights reserved.
% License: BSD 2-Clause license (see COPYING)
%

clc;
clear;
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
fprintf(1, 'Loading butterfly gyroscope.\n')
fprintf(1, '----------------------------------------------\n')

% From: https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Butterfly_Gyroscope
% Rayleigh Damping: D = alpha*M + beta*K
%   alpha = 0;
%   beta  = 1e-6;
load('data/Butterfly.mat')

% Rayleigh damping coefficients.
alpha = 0;
beta  = 1e-6;

%% Part 1.
% Assume true damping coefficients are known.

%% Reduced order models.
% Test performance from i[1e4, 1e6].
a = 4;  b = 6;  nNodes = 200;          

% Prepare quadrature weights and nodes according to Trapezoidal rule.
[nodesLeft, weightsLeft, nodesRight, weightsRight] = trapezoidal_rule([a, b], ...
    nNodes, true);

% Put into complex conjugate pairs to make reduced-order model matrices
% real valued. 
[nodesLeft, Ileft]   = sort(nodesLeft, 'ascend');    
[nodesRight, Iright] = sort(nodesRight, 'ascend');   
weightsLeft          = weightsLeft(Ileft);
weightsRight         = weightsRight(Iright);

% Order of reduction.
r = 10;

% Transfer function data.
recomputeSamples = false;
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
    save('results/Butterfly_Samples_N200_1e4to1e6.mat', 'GsLeft', 'GsRight', ...
        'nodesLeft', 'nodesRight')
else
    fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
    fprintf(1, '-------------------------------------------\n')
    load('results/butterfly_samples_N200_1e4to1e6.mat', 'GsLeft', 'GsRight')
end

% Non-intrusive methods.
%% 1. soQuadBT.
fprintf(1, 'BUILDING LOEWNER MATRICES (soQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;

% Loewner matrices.
[Mbar_soQuadBT, ~, Kbar_soQuadBT, Bbar_soQuadBT, CpBar_soQuadBT] = ...
    so_loewner_factory(nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeft, ...
                       GsRight, 'Rayleigh', [alpha, beta], 'Position');
fprintf(1, 'CONSTRUCTION OF LOEWNER MATRICES FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

% Make it real-valued.
Jp = zeros(nNodes*p, nNodes*p);
Jm = zeros(nNodes*m, nNodes*m);
Ip = eye(p, p);
for i = 1:nNodes/2
    Jp(1 + 2*(i - 1)*p:2*i*p, 1 + 2*(i - 1)*p:2*i*p) = 1/sqrt(2)*[Ip, -1i*Ip; Ip, 1i*Ip];
    Jm(1 + 2*(i - 1):2*i,   1 + 2*(i - 1):2*i)       = 1/sqrt(2)*[1,  -1i;    1,  1i];
end

Mbar_soQuadBT = Jp'*Mbar_soQuadBT*Jm; Kbar_soQuadBT  = Jp'*Kbar_soQuadBT*Jm;   
Mbar_soQuadBT = real(Mbar_soQuadBT);  Kbar_soQuadBT  = real(Kbar_soQuadBT);  
Bbar_soQuadBT = Jp'*Bbar_soQuadBT;    CpBar_soQuadBT = CpBar_soQuadBT*Jm;
Bbar_soQuadBT = real(Bbar_soQuadBT);  CpBar_soQuadBT = real(CpBar_soQuadBT);
Dbar_soQuadBT = alpha*Mbar_soQuadBT + beta*Kbar_soQuadBT;

recomputeModel = false;
if recomputeModel
    fprintf(1, 'COMPUTING REDUCED-ORDER MODEL (soQuadBT).\n')
    fprintf(1, '--------------------------------------\n')
    timeRed = tic;

    % Reductor.
    [Z_soQuadBT, S_soQuadBT, Y_soQuadBT] = svd(Mbar_soQuadBT);

    % Reduced model matrices.
    Mr_soQuadBT  = eye(r, r);
    Kr_soQuadBT  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Kbar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Dr_soQuadBT  = alpha*Mr_soQuadBT + beta*Kr_soQuadBT;
    Cpr_soQuadBT = CpBar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Br_soQuadBT  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Bbar_soQuadBT;
        
    fprintf(1, 'REDUCED-ORDER MODEL COMPUTED IN %.2f s\n', toc(timeRed))
    fprintf(1, '--------------------------------------\n')

    filename = 'results/roButterfly_soQuadBT_r10_N200_1e4to1e6.mat';
    save(filename, 'Mr_soQuadBT', 'Dr_soQuadBT', 'Kr_soQuadBT', 'Br_soQuadBT', 'Cpr_soQuadBT');
else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (soQuadBT).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roButterfly_soQuadBT_r10_N200_1e4to1e6.mat')
end

%% 2. soLoewner.
fprintf(1, 'BUILDING LOEWNER MATRICES (soLoewner).\n')
fprintf(1, '---------------------------------------\n')
timeLoewner = tic;

% Loewner matrices.
[Mbar_soLoewner, ~, Kbar_soLoewner, Bbar_soLoewner, CpBar_soLoewner] = ...
    so_loewner_factory(nodesLeft, nodesRight, ones(nNodes, 1), ones(nNodes, 1), GsLeft, ...
                       GsRight, 'Rayleigh', [alpha, beta], 'Position');
fprintf(1, 'CONSTRUCTION OF LOEWNER MATRICES FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

% Make real valued.
Mbar_soLoewner = Jp'*Mbar_soLoewner*Jm; Kbar_soLoewner  = Jp'*Kbar_soLoewner*Jm;   
Mbar_soLoewner = real(Mbar_soLoewner);  Kbar_soLoewner  = real(Kbar_soLoewner);  
Bbar_soLoewner = Jp'*Bbar_soLoewner;    CpBar_soLoewner = CpBar_soLoewner*Jm;
Bbar_soLoewner = real(Bbar_soLoewner);  CpBar_soLoewner = real(CpBar_soLoewner);
Dbar_soLoewner = alpha*Mbar_soLoewner + beta*Kbar_soLoewner;

recomputeModel = false;
if recomputeModel
    fprintf(1, 'COMPUTING REDUCED-ORDER MODEL (soLoewner).\n')
    fprintf(1, '--------------------------------------\n')
    timeRed = tic;

    % Reductor.
    % Relevant SVDs.
    [Yl_soLoewner, Sl_soLoewner, ~] = svd([-Mbar_soLoewner, Kbar_soLoewner], 'econ');
    [~, Sr_soLoewner, Xr_soLoewner] = svd([-Mbar_soLoewner; Kbar_soLoewner], 'econ');
    
    % Compress.
    Mr_soLoewner  = Yl_soLoewner(:, 1:r)'*Mbar_soLoewner*Xr_soLoewner(:, 1:r); 
    Kr_soLoewner  = Yl_soLoewner(:, 1:r)'*Kbar_soLoewner*Xr_soLoewner(:, 1:r);
    Dr_soLoewner  = alpha*Mr_soLoewner + beta*Kr_soLoewner;
    Br_soLoewner  = Yl_soLoewner(:, 1:r)'*Bbar_soLoewner;
    Cpr_soLoewner = CpBar_soLoewner*Xr_soLoewner(:, 1:r);

    fprintf(1, 'REDUCED-ORDER MODEL COMPUTED IN %.2f s\n', toc(timeRed))
    fprintf(1, '--------------------------------------\n')
    
    filename = 'results/roButterfly_soLoewner_r10_N200_1e4to1e6.mat';
    save(filename, 'Mr_soLoewner', 'Dr_soLoewner', 'Kr_soLoewner', 'Br_soLoewner', 'Cpr_soLoewner');
else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (soLoewner).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roButterfly_soLoewner_r10_N200_1e4to1e6.mat')
end

%% 3. foQuadBT.
fprintf(1, 'BUILDING LOEWNER MATRICES (foQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;
% Loewner matrices.
[Ebar_foQuadBT, Abar_foQuadBT, Bbar_foQuadBT, Cbar_foQuadBT] = fo_loewner_factory(...
    nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeft, GsRight);
fprintf(1, 'CONSTRUCTION OF LOEWNER MATRICES FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

% Make real valued.
Ebar_foQuadBT = Jp'*Ebar_foQuadBT*Jm; Abar_foQuadBT = Jp'*Abar_foQuadBT*Jm;   
Ebar_foQuadBT = real(Ebar_foQuadBT);  Abar_foQuadBT = real(Abar_foQuadBT);  
Bbar_foQuadBT = Jp'*Bbar_foQuadBT;    Cbar_foQuadBT = Cbar_foQuadBT*Jm;
Bbar_foQuadBT = real(Bbar_foQuadBT);  Cbar_foQuadBT = real(Cbar_foQuadBT);

recomputeModel = false;
if recomputeModel
    fprintf(1, 'COMPUTING REDUCED-ORDER MODEL (foQuadBT).\n')
    fprintf(1, '--------------------------------------\n')
    timeRed = tic;

    % Reductor.
    [Z_foQuadBT, S_foQuadBT, Y_foQuadBT] = svd(Ebar_foQuadBT);

    % Reduced model matrices.
    Er_foQuadBT  = eye(r, r);
    Ar_foQuadBT  = (S_foQuadBT(1:r, 1:r)^(-1/2)*Z_foQuadBT(:, 1:r)')*Abar_foQuadBT*(Y_foQuadBT(:, 1:r)*S_foQuadBT(1:r, 1:r)^(-1/2));
    Cr_foQuadBT  = Cbar_foQuadBT*(Y_foQuadBT(:, 1:r)*S_foQuadBT(1:r, 1:r)^(-1/2));
    Br_foQuadBT  = (S_foQuadBT(1:r, 1:r)^(-1/2)*Z_foQuadBT(:, 1:r)')*Bbar_foQuadBT;

    fprintf(1, 'REDUCED-ORDER MODEL COMPUTED IN %.2f s\n', toc(timeRed))
    fprintf(1, '--------------------------------------\n')
    
    filename = 'results/roButterfly_foQuadBT_r10_N200_1e4to1e6.mat';
    save(filename, 'Er_foQuadBT', 'Ar_foQuadBT', 'Br_foQuadBT', 'Cr_foQuadBT');

else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (foQuadBT).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roButterfly_foQuadBT_r10_N200_1e4to1e6.mat')
end

% Intrusive methods.
%% 4. soBT.
recomputeModel = false;
if recomputeModel
    fprintf(1, 'COMPUTING REDUCED-ORDER MODEL (soBT).\n')
    fprintf(1, '--------------------------------------\n')
    timeRed = tic;

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
    
    [soBTRom, info_soBT] = ml_ct_s_soss_bt(soSys, opts);
    Mr_soBT  = soBTRom.M;
    Dr_soBT  = soBTRom.E;
    Kr_soBT  = soBTRom.K;
    Br_soBT  = soBTRom.Bu;
    Cpr_soBT = soBTRom.Cp;

    fprintf(1, 'REDUCED-ORDER MODEL COMPUTED IN %.2f s\n', toc(timeRed))
    fprintf(1, '--------------------------------------\n')
    
    filename = 'results/roButterfly_soBT_r10.mat';
    save(filename, 'Mr_soBT', 'Dr_soBT', 'Kr_soBT', 'Br_soBT', 'Cpr_soBT');
else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (soBT).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roButterfly_soBT_r10.mat')
end

%% 5. foBt.
recomputeModel = false;
if recomputeModel
    fprintf(1, 'COMPUTING REDUCED-ORDER MODEL (foBT).\n')
    fprintf(1, '--------------------------------------\n')
    timeRed = tic;

    % Lift realization for projection.
    Efo                   = spalloc(2*n, 2*n, nnz(M) + n); % Descriptor matrix; Efo = [I, 0: 0, M]
    Efo(1:n, 1:n)         = speye(n);                      % (1, 1) block
    Efo(n+1:2*n, n+1:2*n) = M;                             % (2, 2) block is (sparse) mass matrix M
    
    Afo                   = spalloc(2*n, 2*n, nnz(K) + nnz(D) + n); % Afo = [0, I; -K, -D]
    Afo(1:n, n+1:2*n)     = speye(n);                               % (1, 2) block of Afo
    Afo(n+1:2*n, 1:n)     = -K;                                     % (2, 1) block is -K
    Afo(n+1:2*n, n+1:2*n) = -D;                                     % (2, 2) block is -D 
    
    Bfo             = spalloc(2*n, 1, nnz(B)); % Bfo = [0; B];
    Bfo(n+1:2*n, :) = B; 

    Cfo         = spalloc(p, 2*n, nnz(C)); % Bfo = [Cp, 0];
    Cfo(:, 1:n) = C; 

    foSys   = struct();
    foSys.E = Efo;
    foSys.A = Afo;
    foSys.B = Bfo;
    foSys.C = Cfo;
    foSys.D = zeros(p, m);
    
    % Input opts.
    opts                  = struct();
    opts.Order            = r;
    opts.OrderComputation = 'order';
    
    [foBTRom, info_foBT] = ml_ct_s_foss_bt(foSys, opts);
    Er_foBT         = foBTRom.E;
    Ar_foBT         = foBTRom.A;
    Br_foBT         = foBTRom.B;
    Cr_foBT         = foBTRom.C;

    fprintf(1, 'REDUCED-ORDER MODEL COMPUTED IN %.2f s\n', toc(timeRed))
    fprintf(1, '--------------------------------------\n')
    
    filename = 'results/roButterfly_foBT_r10.mat';
    save(filename, 'Er_foBT', 'Ar_foBT', 'Br_foBT', 'Cr_foBT');
else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (foBT).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roButterfly_foBT_r10.mat')
end

%% Plots.
numSamples = 500;
s          = 1i*logspace(4, 6, numSamples);

% Transfer function evaluations.
Gr_soQuadBT  = zeros(p, m, numSamples);
Gr_soLoewner = zeros(p, m, numSamples);
Gr_foQuadBT  = zeros(p, m, numSamples);
Gr_soBT      = zeros(p, m, numSamples);
Gr_foBT      = zeros(p, m, numSamples);

% Magnitude response and errors.
resp_soQuadBT          = zeros(numSamples, 1); % Response of (non-intrusive) soQuadBT reduced model
relSVError_soQuadBT    = zeros(numSamples, 1); % Rel. SV error due to (non-intrusive) soQuadBT reduced model
absSVError_soQuadBT    = zeros(numSamples, 1); % Abs. SV error due to (non-intrusive) soQuadBT reduced model
absFrobError_soQuadBT  = zeros(numSamples, 1); % Abs. Frob. error due to (non-intrusive) soQuadBT reduced model

resp_soLoewner         = zeros(numSamples, 1); % Response of (non-intrusive) soLoewner reduced model
relSVError_soLoewner   = zeros(numSamples, 1); % Rel. SV error due to (non-intrusive) soLoewner reduced model
absSVError_soLoewner   = zeros(numSamples, 1); % Abs. SV error due to (non-intrusive) soLoewner reduced modell
absFrobError_soLoewner = zeros(numSamples, 1); % Abs. Frob. error due to (non-intrusive) soLoewner reduced model

resp_foQuadBT          = zeros(numSamples, 1); % Response of (non-intrusive) foQuadBT reduced model
relSVError_foQuadBT    = zeros(numSamples, 1); % Rel. SV error due to (non-intrusive) foQuadBT reduced model
absSVError_foQuadBT    = zeros(numSamples, 1); % Abs. SV error due to (non-intrusive) foQuadBT reduced model
absFrobError_foQuadBT  = zeros(numSamples, 1); % Abs. Frob. error due to (non-intrusive) foQuadBT reduced model

resp_soBT              = zeros(numSamples, 1); % Response of (intrusive) soBT reduced model
relSVError_soBT        = zeros(numSamples, 1); % Rel. SV error due to (intrusive) soBT reduced model
absSVError_soBT        = zeros(numSamples, 1); % Abs. SV error due to (intrusive) soBT reduced model
absFrobError_soBT      = zeros(numSamples, 1); % Abs. Frob. error due to (intrusive) soBT reduced model

resp_foBT              = zeros(numSamples, 1); % Response of (intrusive) foBT reduced model
relSVError_foBT        = zeros(numSamples, 1); % Rel. SV error due to (intrusive) foBT reduced model
absSVError_foBT        = zeros(numSamples, 1); % Abs. SV error due to (intrusive) foBT reduced model
absFrobError_foBT      = zeros(numSamples, 1); % Abs. Frob. error due to (intrusive) foBT reduced model

% Full-order simulation data.
recompute = false;
if recompute
    Gfo     = zeros(p, m, numSamples);
    GfoResp = zeros(numSamples, 1);
    GfoFrob = zeros(numSamples, 1);
    fprintf(1, 'Sampling full-order transfer function along i[1e4, 1e6].\n')
    fprintf(1, '--------------------------------------------------------\n')
    for ii = 1:numSamples
        timeSolve = tic;
        fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, imag(s(ii)))
        Gfo(:, :, ii) = C*((s(ii)^2*M +s(ii)*D + K)\B);
        GfoResp(ii)   = max(svd(Gfo(:, :, ii)));        % Matrix 2-norm
        GfoFrob(ii)   = norm(Gfo(:, :, ii), 'fro');     % Matrix Frobenius-norm
        fprintf(1, 'k = %d solve finished in %.2f s\n', ii, toc(timeSolve))
    end
    save('results/butterfly_samples_N500_1e4to1e6.mat', 'Gfo', 'GfoResp', 'GfoFrob')
else
    fprintf(1, 'Loading precomputed values.\n')
    fprintf(1, '--------------------------------------------------------\n')
    load('results/butterfly_samples_N500_1e4to1e6.mat')
end

% Compute frequency response along imaginary axis.
for ii=1:numSamples
    % Transfer functions.
    Gr_soQuadBT(:, :, ii)  = Cpr_soQuadBT*((s(ii)^2*Mr_soQuadBT + s(ii)*Dr_soQuadBT + Kr_soQuadBT)\Br_soQuadBT);
    Gr_soLoewner(:, :, ii) = Cpr_soLoewner*((s(ii)^2*Mr_soLoewner + s(ii)*Dr_soLoewner + Kr_soLoewner)\Br_soLoewner);
    Gr_foQuadBT(:, :, ii)  = Cr_foQuadBT*((s(ii)*Er_foQuadBT - Ar_foQuadBT)\Br_foQuadBT);
    Gr_soBT(:, :, ii)      = Cpr_soBT*((s(ii)^2*Mr_soBT + s(ii)*Dr_soBT + Kr_soBT)\Br_soBT);
    Gr_foBT(:, :, ii)      = Cr_foBT*((s(ii)*Er_foBT - Ar_foBT)\Br_foBT);

    % Response and errors. 
    resp_soQuadBT(ii)          = max(svd(Gr_soQuadBT(:, :, ii))); 
    absSVError_soQuadBT(ii)    = max(svd(Gfo(:, :, ii) - Gr_soQuadBT(:, :, ii)));
    relSVError_soQuadBT(ii)    = absSVError_soQuadBT(ii)/GfoResp(ii);
    absFrobError_soQuadBT(ii)  = norm((Gfo(:, :, ii) - Gr_soQuadBT(:, :, ii)), 'fro');

    resp_soLoewner(ii)         = max(svd(Gr_soLoewner(:, :, ii))); 
    absSVError_soLoewner(ii)   = max(svd(Gfo(:, :, ii) - Gr_soLoewner(:, :, ii)));
    relSVError_soLoewner(ii)   = absSVError_soLoewner(ii)/GfoResp(ii); 
    absFrobError_soLoewner(ii) = norm((Gfo(:, :, ii) - Gr_soLoewner(:, :, ii)), 'fro');

    resp_foQuadBT(ii)          = max(svd(Gr_foQuadBT(:, :, ii))); 
    absSVError_foQuadBT(ii)    = max(svd(Gfo(:, :, ii) - Gr_foQuadBT(:, :, ii))); 
    relSVError_foQuadBT(ii)    = absSVError_foQuadBT(ii)/GfoResp(ii); 
    absFrobError_foQuadBT(ii)  = norm((Gfo(:, :, ii) - Gr_foQuadBT(:, :, ii)), 'fro');

    resp_soBT(ii)              = max(svd(Gr_soBT(:, :, ii))); 
    absSVError_soBT(ii)        = max(svd(Gfo(:, :, ii) - Gr_soBT(:, :, ii)));
    relSVError_soBT(ii)        = absSVError_soBT(ii)/GfoResp(ii); 
    absFrobError_soBT(ii)      = norm((Gfo(:, :, ii) - Gr_soBT(:, :, ii)), 'fro');

    resp_foBT(ii)              = max(svd(Gr_foBT(:, :, ii))); 
    absSVError_foBT(ii)        = max(svd(Gfo(:, :, ii) - Gr_foBT(:, :, ii)));
    relSVError_foBT(ii)        = absSVError_foBT(ii)/GfoResp(ii); 
    absFrobError_foBT(ii)      = norm((Gfo(:, :, ii) - Gr_foBT(:, :, ii)), 'fro');
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
    ColMat(6,:) = [1         0.4       0.6];
    
    figure(1)
    fs = 12;
    % Magnitudes
    set(gca, 'fontsize', 10)
    subplot(2,1,1)
    loglog(imag(s), GfoResp,        '-o',  'linewidth', 2, 'color', ColMat(1,:)); hold on
    loglog(imag(s), resp_soQuadBT,  '--',  'linewidth', 2, 'color', ColMat(2,:)); 
    loglog(imag(s), resp_soLoewner, '--',  'linewidth', 2, 'color', ColMat(3,:)); 
    loglog(imag(s), resp_foQuadBT,  '-.',  'linewidth', 2, 'color', ColMat(4,:)); 
    loglog(imag(s), resp_soBT,      '--.', 'linewidth', 2, 'color', ColMat(5,:)); 
    loglog(imag(s), resp_foBT,      '-.', 'linewidth', 2, 'color',  ColMat(6,:)); 

    leg = legend('Full-order', 'soQuadBT', 'soLoewner', 'foQuadBT', 'soBT', 'foBT', ...
        'location', 'southeast', 'orientation', 'horizontal', 'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')

    xlabel('$i*\omega$',            'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')
    
    % Relative errors
    subplot(2,1,2)
    loglog(imag(s), relSVError_soQuadBT,  '-o', 'linewidth', 2, 'color', ColMat(2,:)); hold on
    loglog(imag(s), relSVError_soLoewner, '-*', 'linewidth', 2, 'color', ColMat(3,:));
    loglog(imag(s), relSVError_foQuadBT,  '-*', 'linewidth', 2, 'color', ColMat(4,:));
    loglog(imag(s), relSVError_soBT,      '-*', 'linewidth', 2, 'color', ColMat(5,:));
    loglog(imag(s), relSVError_foBT,      '-*', 'linewidth', 2, 'color', ColMat(6,:));

    leg = legend('soQuadBT', 'soLoewner', 'foQuadBT', 'soBT', 'foBT', 'location', ...
        'southeast', 'orientation', 'horizontal', 'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')

    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', ...
        fs, 'interpreter', 'latex')
end

% Store data.
write = true;
if write
    magMatrix = [imag(s)', GfoResp, resp_soQuadBT, resp_soLoewner, resp_foQuadBT, ...
        resp_soBT, resp_foBT];
    dlmwrite('results/butterfly_r10_N200_1e4to1e6_mag.dat', magMatrix, ...
        'delimiter', '\t', 'precision', 8);
    errorMatrix = [imag(s)', relSVError_soQuadBT, relSVError_soLoewner, relSVError_foQuadBT ...
        relSVError_soBT, relSVError_foBT];
    dlmwrite('results/butterfly_r10_N200_1e4to1e6_error.dat', errorMatrix, ...
        'delimiter', '\t', 'precision', 8);
end

%% Error measures.
% Print errors.
fprintf(1, 'Order r = %d.\n', r)
fprintf(1, '--------------\n')
fprintf(1, 'Relative H-infty error due to soQuadBT : %.16f \n', max((absSVError_soQuadBT))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to soLoewner: %.16f \n', max((absSVError_soLoewner))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to foQuadBT : %.16f \n', max((absSVError_foQuadBT))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to soBT     : %.16f \n', max((absSVError_soBT))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to foBT     : %.16f \n', max((absSVError_foBT))./max((GfoResp)))
fprintf(1, '------------------------------------------------------------\n')
fprintf(1, 'Relative H-2 error due to soQuadBT     : %.16f \n', sqrt(sum(absFrobError_soQuadBT.^2)/sum(GfoFrob.^2)))
fprintf(1, 'Relative H-2 error due to soLoewner    : %.16f \n', sqrt(sum(absFrobError_soLoewner.^2)/sum(GfoFrob.^2)))
fprintf(1, 'Relative H-2 error due to foQuadBT     : %.16f \n', sqrt(sum(absFrobError_foQuadBT.^2)/sum(GfoFrob.^2)))
fprintf(1, 'Relative H-2 error due to soBT         : %.16f \n', sqrt(sum(absFrobError_soBT.^2)/sum(GfoFrob.^2)))
fprintf(1, 'Relative H-2 error due to foBT         : %.16f \n', sqrt(sum(absFrobError_foBT.^2)/sum(GfoFrob.^2)))
fprintf(1, '------------------------------------------------------------\n')

%% Part 2.
% Estimate damping coefficients from data.

%%
% Options for fminunc.
options = optimoptions('fminunc', 'Display', 'iter-detailed', 'Algorithm', 'quasi-newton', ...
    'SpecifyObjectiveGradient', true, 'OptimalityTolerance', 1e-8, 'StepTolerance', 1e-8);

% Test three different pairs of initial values for optimization variables.
alpha1 = 1e-4;    beta1  = 1e-4;
alpha2 = 0;       beta2  = 0;
alpha3 = 1e-1;    beta3  = 1e-1;

initDampingParams1 = [alpha1, beta1];
initDampingParams2 = [alpha2, beta2];
initDampingParams3 = [alpha3, beta3];

% Instantiate objective function to pass to solver. 
objFunc = @(dampingParams) rayleigh_damping_obj(dampingParams, [nodesLeft; nodesRight], ...
    [cat(3, GsLeft, GsRight)], Kr_soQuadBT, Br_soQuadBT, Cpr_soQuadBT, zeros(p, r));

% Minimizer.
optDampingParams1 = fminunc(objFunc, initDampingParams1, options);
optDampingParams2 = fminunc(objFunc, initDampingParams2, options);
optDampingParams3 = fminunc(objFunc, initDampingParams3, options);

% Found parameters.
optAlpha1 = optDampingParams1(1); optBeta1 = optDampingParams1(2);
optAlpha2 = optDampingParams2(1); optBeta2 = optDampingParams2(2);
optAlpha3 = optDampingParams3(1); optBeta3 = optDampingParams3(2);

fprintf(1, 'FOUND DAMPING PARAMETERS (Initialization 1).\n')
fprintf(1, 'optAlpha1          : %.16f\n', optAlpha1)
fprintf(1, 'optbeta1           : %.16f\n', optBeta1)
fprintf(1, '--------------------------------------------------------\n')
fprintf(1, 'DIFFERENCE IN FOUND DAMPING PARAMETERS COMPARED TO TRUE (Initialization 1).\n')
fprintf(1, '|alpha - optAlpha1|: %.16f\n', abs(alpha - optAlpha1))
fprintf(1, '|beta  - optbeta1 |: %.16f\n', abs(beta  - optBeta1))
fprintf(1, '--------------------------------------------------------\n')

fprintf(1, 'FOUND DAMPING PARAMETERS (Initialization 2).\n')
fprintf(1, 'optAlpha2          : %.16f\n', optAlpha2)
fprintf(1, 'optbeta2           : %.16f\n', optBeta2)
fprintf(1, '--------------------------------------------------------\n')
fprintf(1, 'DIFFERENCE IN FOUND DAMPING PARAMETERS COMPARED TO TRUE (Initialization 2).\n')
fprintf(1, '|alpha - optAlpha2|: %.16f\n', abs(alpha - optAlpha2))
fprintf(1, '|beta  - optbeta2 |: %.16f\n', abs(beta  - optBeta2))
fprintf(1, '--------------------------------------------------------\n')

fprintf(1, 'FOUND DAMPING PARAMETERS (Initialization 3).\n')
fprintf(1, 'optAlpha3          : %.16f\n', optAlpha3)
fprintf(1, 'optbeta3           : %.16f\n', optBeta3)
fprintf(1, '--------------------------------------------------------\n')
fprintf(1, 'DIFFERENCE IN FOUND DAMPING PARAMETERS COMPARED TO TRUE (Initialization 3).\n')
fprintf(1, '|alpha - optAlpha3|: %.16f\n', abs(alpha - optAlpha3))
fprintf(1, '|beta  - optbeta3 |: %.16f\n', abs(beta  - optBeta3))
fprintf(1, '--------------------------------------------------------\n')

% Damping with inferred parameters. 
Dr_soQuadBT_optParams1 = optAlpha1*Mr_soQuadBT + optBeta1*Kr_soQuadBT;
Dr_soQuadBT_optParams2 = optAlpha2*Mr_soQuadBT + optBeta2*Kr_soQuadBT;
Dr_soQuadBT_optParams3 = optAlpha3*Mr_soQuadBT + optBeta3*Kr_soQuadBT;

%% Plot response of reduced models.
% Transfer function evaluations.
Gr_soQuadBT_optParams1 = zeros(p, m, numSamples);
Gr_soQuadBT_optParams2 = zeros(p, m, numSamples);
Gr_soQuadBT_optParams3 = zeros(p, m, numSamples);

% Response and errors.
resp_soQuadBT_optParams1       = zeros(numSamples, 1); % Response of reduced model 1
relSVError_soQuadBT_optParams1 = zeros(numSamples, 1); % Error due to reduced model 1

resp_soQuadBT_optParams2       = zeros(numSamples, 1); % Response of reduced model 2
relSVError_soQuadBT_optParams2 = zeros(numSamples, 1); % Error due to reduced model 2

resp_soQuadBT_optParams3       = zeros(numSamples, 1); % Response of reduced model 3
relSVError_soQuadBT_optParams3 = zeros(numSamples, 1); % Error due to reduced model 3

% Response of full-order and soQuadBT reduced-order already computed.
% Compute frequency response along imaginary axis.
for ii=1:numSamples
    % Transfer functions.
    Gr_soQuadBT_optParams1 = Cpr_soQuadBT*((s(ii)^2*Mr_soQuadBT + s(ii)*Dr_soQuadBT_optParams1 + Kr_soQuadBT)\Br_soQuadBT);
    Gr_soQuadBT_optParams2 = Cpr_soQuadBT*((s(ii)^2*Mr_soQuadBT + s(ii)*Dr_soQuadBT_optParams2 + Kr_soQuadBT)\Br_soQuadBT);
    Gr_soQuadBT_optParams3 = Cpr_soQuadBT*((s(ii)^2*Mr_soQuadBT + s(ii)*Dr_soQuadBT_optParams3 + Kr_soQuadBT)\Br_soQuadBT);

    resp_soQuadBT_optParams1(ii) = max(svd(Gr_soQuadBT_optParams1));
    resp_soQuadBT_optParams2(ii) = max(svd(Gr_soQuadBT_optParams2));
    resp_soQuadBT_optParams3(ii) = max(svd(Gr_soQuadBT_optParams3));

    relSVError_soQuadBT_optParams1(ii) = max(svd(Gfo(:, :, ii) - Gr_soQuadBT_optParams1))/GfoResp(ii);
    relSVError_soQuadBT_optParams2(ii) = max(svd(Gfo(:, :, ii) - Gr_soQuadBT_optParams2))/GfoResp(ii);
    relSVError_soQuadBT_optParams3(ii) = max(svd(Gfo(:, :, ii) - Gr_soQuadBT_optParams3))/GfoResp(ii);
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
    ColMat(6,:) = [1         0.4       0.6];
    
    figure
    fs = 12;
    % Magnitudes.
    set(gca, 'fontsize', 10)
    subplot(2,1,1)
    loglog(imag(s), GfoResp,                  '-o', 'linewidth', 2, 'color', ColMat(1,:)); hold on
    loglog(imag(s), resp_soQuadBT,            '--', 'linewidth', 2, 'color', ColMat(2,:)); 
    loglog(imag(s), resp_soQuadBT_optParams1, '-.', 'linewidth', 2, 'color', ColMat(3,:)); 
    loglog(imag(s), resp_soQuadBT_optParams2, '-.', 'linewidth', 2, 'color', ColMat(4,:)); 
    loglog(imag(s), resp_soQuadBT_optParams3, '-.', 'linewidth', 2, 'color', ColMat(5,:)); 
    leg = legend('Full-order', 'soQuadBT (true damping)', 'soQuadBT (optimal/computed damping 1)', ...
         'soQuadBT (optimal/computed damping 2)',  'soQuadBT (optimal/computed damping 3)', ...
         'location', 'southeast', 'orientation', 'horizontal', 'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')
    
    % Relative errors.
    subplot(2,1,2)
    loglog(imag(s), relSVError_soQuadBT,            '-o', 'linewidth', 2, 'color', ColMat(2,:)); hold on
    loglog(imag(s), relSVError_soQuadBT_optParams1, '-.', 'linewidth', 2, 'color', ColMat(3,:)); 
    loglog(imag(s), relSVError_soQuadBT_optParams2, '-.', 'linewidth', 2, 'color', ColMat(4,:)); 
    loglog(imag(s), relSVError_soQuadBT_optParams3, '-.', 'linewidth', 2, 'color', ColMat(5,:)); 
    eg = legend('soQuadBT (true damping)', 'soQuadBT (optimal/computed damping 1)', ...
         'soQuadBT (optimal/computed damping 2)',  'soQuadBT (optimal/computed damping 3)', ...
         'location', 'southeast', 'orientation', 'horizontal', 'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', ...
        fs, 'interpreter', 'latex')
end

% Store data.
write = true;
if write
    magMatrix = [imag(s)', GfoResp, resp_soQuadBT, resp_soQuadBT_optParams1, ...
        resp_soQuadBT_optParams2, resp_soQuadBT_optParams3];
    dlmwrite('results/butterfly_dampingOpt_r10_N200_1e4to1e6_mag.dat', magMatrix, ...
        'delimiter', '\t', 'precision', 8);
    errorMatrix = [imag(s)', relSVError_soQuadBT, relSVError_soQuadBT_optParams1, ...
        relSVError_soQuadBT_optParams2, relSVError_soQuadBT_optParams3];
    dlmwrite('results/butterfly_dampingOpt_r10_N200_1e4to1e6_error.dat', errorMatrix, ...
        'delimiter', '\t', 'precision', 8);
end

%% Part 3.
% Symmetric system, second-order Hermite Loewner matrices.

% Cp models average displacement of electrode 1 in x-, y-, and
% z-directions; input matrix is taken to be Cp'.
Cp          = spalloc(1, n, 3); 
Cp(1, 3295) = 1/n;
Cp(1, 3296) = 1/n;
Cp(1, 3297) = 1/n;
B           = Cp';
p           = 1;

% Recompute nodes. 
% Prepare quadrature weights and nodes according to Trapezoidal rule.
[nodes, weights, ~, ~] = trapezoidal_rule([a, b], nNodes, false);

% Put into complex conjugate pairs to make reduced-order model matrices
% real valued. 
[nodes, I] = sort(nodes, 'ascend');    
weights    = weights(I);

% Order of reduction.
r = 20;

GsLeft_singleOut  = zeros(p, m, nNodes);
GsRight_singleOut = zeros(p, m, nNodes);
Gs_singleOut      = zeros(p, m, nNodes);
GsDeriv_singleOut = zeros(p, m, nNodes);

% Transfer function derivatievs.
recomputeSamples = true;
if recomputeSamples
    fprintf(1, 'COMPUTING TRANSFER FUNCTION DATA.\n')
    fprintf(1, '---------------------------------\n')
    % Space allocation.
    Gs      = zeros(p, m, nNodes);
    GsDeriv = zeros(p, m, nNodes);
    for k = 1:nNodes
        % Requisite linear solves.
        tic
        fprintf(1, 'Linear solve %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % Transfer function data.
        % nodesLeft, nodesRight, saved from earlier.
        GsLeft_singleOut(:, :, k)  = Cp*((nodesLeft(k)^2.*M + nodesLeft(k).*D + K)\B);
        GsRight_singleOut(:, :, k) = Cp*((nodesRight(k)^2.*M + nodesRight(k).*D + K)\B);
        Gs_singleOut(:, :, k)      = Cp*((nodes(k)^2.*M + nodes(k).*D + K)\B);
        GsDeriv_singleOut(:, :, k) = -Cp*((nodes(k)^2.*M + nodes(k).*D + K)\((2*nodes(k)*M + D) ...
            *((nodes(k)^2.*M + nodes(k).*D + K)\B)));
        fprintf(1, 'Solves finished in %.2f s.\n',toc)
        fprintf(1, '-----------------------------\n');
    end
    save('results/butterfly_deriv_samples_N200_1e4to1e6.mat', 'Gs_singleOut', 'GsDeriv_singleOut', ...
        'GsLeft_singleOut', 'GsRight_singleOut', 'nodes')
else
    fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
    fprintf(1, '-------------------------------------------\n')
    load('results/butterfly_deriv_samples_N200_1e4to1e6.mat', 'Gs_singleOut' ,'GsDeriv_singleOut', ...
        'GsLeft_singleOut', 'GsRight_singleOut')
end

%% Reduced-order models.
%% 1. soQuadBT.
fprintf(1, 'BUILDING LOEWNER MATRICES (soQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;

% Loewner matrices.
[Mbar_soQuadBT, ~, Kbar_soQuadBT, Bbar_soQuadBT, CpBar_soQuadBT] = ...
    so_loewner_factory(nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeft_singleOut, ...
                       GsRight_singleOut, 'Rayleigh', [alpha, beta], 'Position');
fprintf(1, 'CONSTRUCTION OF LOEWNER MATRICES FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

% Make it real-valued.
Jp = zeros(nNodes*p, nNodes*p);
Jm = zeros(nNodes*m, nNodes*m);
Ip = eye(p, p);
for i = 1:nNodes/2
    Jp(1 + 2*(i - 1)*p:2*i*p, 1 + 2*(i - 1)*p:2*i*p) = 1/sqrt(2)*[Ip, -1i*Ip; Ip, 1i*Ip];
    Jm(1 + 2*(i - 1):2*i,   1 + 2*(i - 1):2*i)       = 1/sqrt(2)*[1,  -1i;    1,  1i];
end

Mbar_soQuadBT = Jp'*Mbar_soQuadBT*Jm; Kbar_soQuadBT  = Jp'*Kbar_soQuadBT*Jm;   
Mbar_soQuadBT = real(Mbar_soQuadBT);  Kbar_soQuadBT  = real(Kbar_soQuadBT);  
Bbar_soQuadBT = Jp'*Bbar_soQuadBT;    CpBar_soQuadBT = CpBar_soQuadBT*Jm;
Bbar_soQuadBT = real(Bbar_soQuadBT);  CpBar_soQuadBT = real(CpBar_soQuadBT);
Dbar_soQuadBT = alpha*Mbar_soQuadBT + beta*Kbar_soQuadBT;

recomputeModel = true;
if recomputeModel
    fprintf(1, 'COMPUTING REDUCED-ORDER MODEL (soQuadBT).\n')
    fprintf(1, '--------------------------------------\n')
    timeRed = tic;

    % Reductor.
    [Z_soQuadBT, S_soQuadBT, Y_soQuadBT] = svd(Mbar_soQuadBT);

    % Reduced model matrices.
    Mr_soQuadBT_singleOut  = eye(r, r);
    Kr_soQuadBT_singleOut  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Kbar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Dr_soQuadBT_singleOut  = alpha*Mr_soQuadBT_singleOut + beta*Kr_soQuadBT_singleOut;
    Cpr_soQuadBT_singleOut = CpBar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Br_soQuadBT_singleOut  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Bbar_soQuadBT;
        
    fprintf(1, 'REDUCED-ORDER MODEL COMPUTED IN %.2f s\n', toc(timeRed))
    fprintf(1, '--------------------------------------\n')

    filename = 'results/roButterfly_soQuadBT_singleOut_r10_N200_1e4to1e6.mat';
    save(filename, 'Mr_soQuadBT_singleOut', 'Dr_soQuadBT_singleOut', 'Kr_soQuadBT_singleOut', 'Br_soQuadBT_singleOut', 'Cpr_soQuadBT_singleOut');
else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (soQuadBT).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roButterfly_soQuadBT_singleOut_r10_N200_1e4to1e6.mat')
end

%% 2. soQuadBT (Hermite matrices).
fprintf(1, 'BUILDING HERMITE LOEWNER MATRICES (soQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;

% Loewner matrices.
[Mbar_soQuadBT_Hermite, ~, Kbar_soQuadBT_Hermite, Bbar_soQuadBT_Hermite, CpBar_soQuadBT_Hermite] = ...
    so_hermite_loewner_factory(nodes, weights, Gs_singleOut, GsDeriv_singleOut, 'Rayleigh', [alpha, beta]);
fprintf(1, 'CONSTRUCTION OF LOEWNER MATRICES FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

Mbar_soQuadBT_Hermite = Jp'*Mbar_soQuadBT_Hermite*Jm; Kbar_soQuadBT_Hermite  = Jp'*Kbar_soQuadBT_Hermite*Jm;   
Mbar_soQuadBT_Hermite = real(Mbar_soQuadBT_Hermite);  Kbar_soQuadBT_Hermite  = real(Kbar_soQuadBT_Hermite);  
Bbar_soQuadBT_Hermite = Jp'*Bbar_soQuadBT_Hermite;    CpBar_soQuadBT_Hermite = CpBar_soQuadBT_Hermite*Jm;
Bbar_soQuadBT_Hermite = real(Bbar_soQuadBT_Hermite);  CpBar_soQuadBT_Hermite = real(CpBar_soQuadBT_Hermite);
Dbar_soQuadBT = alpha*Mbar_soQuadBT_Hermite + beta*Kbar_soQuadBT_Hermite;

recomputeModel = true;
if recomputeModel
    fprintf(1, 'COMPUTING REDUCED-ORDER MODEL (soQuadBT).\n')
    fprintf(1, '--------------------------------------\n')
    timeRed = tic;

    % Reductor.
    [Z_soQuadBT, S_soQuadBT, Y_soQuadBT] = svd(Mbar_soQuadBT_Hermite);

    % Reduced model matrices.
    Mr_soQuadBT_Hermite  = eye(r, r);
    Kr_soQuadBT_Hermite  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Kbar_soQuadBT_Hermite*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Dr_soQuadBT_Hermite  = alpha*Mr_soQuadBT_Hermite + beta*Kr_soQuadBT_Hermite;
    Cpr_soQuadBT_Hermite = CpBar_soQuadBT_Hermite*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Br_soQuadBT_Hermite  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Bbar_soQuadBT_Hermite;
        
    fprintf(1, 'REDUCED-ORDER MODEL COMPUTED IN %.2f s\n', toc(timeRed))
    fprintf(1, '--------------------------------------\n')

    filename = 'results/roButterfly_soQuadBT_Hermite_r10_N200_1e4to1e6.mat';
    save(filename, 'Mr_soQuadBT_Hermite', 'Dr_soQuadBT_Hermite', 'Kr_soQuadBT_Hermite', 'Br_soQuadBT_Hermite', 'Cpr_soQuadBT_Hermite');
else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (soQuadBT).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roButterfly_soQuadBT_Hermite_r10_N200_1e4to1e6.mat')
end

%% Plot response of reduced models.
% Transfer function evaluations.
Gr_soQuadBT_Hermite = zeros(p, m, numSamples);

% Response and errors.
resp_soQuadBT               = zeros(numSamples, 1); % Response of soQuadBT reduced model with usual Loewner matrices
relSVError_soQuadBT         = zeros(numSamples, 1); % Error due to soQuadBT reduced model with usual Loewner matrices

resp_soQuadBT_Hermite       = zeros(numSamples, 1); % Response of soQuadBT reduced model with Hermite Loewner matrices
relSVError_soQuadBT_Hermite = zeros(numSamples, 1); % Error due to soQuadBT reduced model with Hermite Loewner matrices

% Recompute full-order simulation data for new output.
recompute = false;
if recompute
    Gfo     = zeros(p, m, numSamples);
    GfoResp = zeros(numSamples, 1);
    fprintf(1, 'Sampling full-order transfer function along i[1e4, 1e6].\n')
    fprintf(1, '--------------------------------------------------------\n')
    for ii = 1:numSamples
        timeSolve = tic;
        fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, imag(s(ii)))
        Gfo(:, :, ii) = Cp*((s(ii)^2*M +s(ii)*D + K)\B);
        GfoResp(ii)   = max(svd(Gfo(:, :, ii)));        % Matrix 2-norm
        fprintf(1, 'k = %d solve finished in %.2f s\n', ii, toc(timeSolve))
    end
    save('results/butterfly_deriv_samples_N500_1e4to1e6.mat', 'Gfo', 'GfoResp')
else
    fprintf(1, 'Loading precomputed values.\n')
    fprintf(1, '--------------------------------------------------------\n')
    load('results/butterfly_deriv_samples_N500_1e4to1e6.mat')
end

% Response of full-order and soQuadBT reduced-order already computed.
% Compute frequency response along imaginary axis.
for ii=1:numSamples
    % Transfer functions.
    Gr_soQuadBT                     = Cpr_soQuadBT_singleOut*((s(ii)^2*Mr_soQuadBT_singleOut + s(ii)*Dr_soQuadBT_singleOut + Kr_soQuadBT_singleOut)\Br_soQuadBT_singleOut);
    Gr_soQuadBT_Hermite             = Cpr_soQuadBT_Hermite*((s(ii)^2*Mr_soQuadBT_Hermite ...
        + s(ii)*Dr_soQuadBT_Hermite + Kr_soQuadBT_Hermite)\Br_soQuadBT_Hermite);

    resp_soQuadBT(ii)               = abs(Gr_soQuadBT);
    resp_soQuadBT_Hermite(ii)       = abs(Gr_soQuadBT_Hermite);

    relSVError_soQuadBT(ii)         = abs(Gfo(:, :, ii) - Gr_soQuadBT)/GfoResp(ii);
    relSVError_soQuadBT_Hermite(ii) = abs(Gfo(:, :, ii) - Gr_soQuadBT_Hermite)/GfoResp(ii);
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
    ColMat(6,:) = [1         0.4       0.6];
    
    figure
    fs = 12;
    % Magnitudes.
    set(gca, 'fontsize', 10)
    subplot(2,1,1)
    loglog(imag(s), GfoResp,               '-o', 'linewidth', 2, 'color', ColMat(1,:)); hold on
    loglog(imag(s), resp_soQuadBT,         '--', 'linewidth', 2, 'color', ColMat(2,:)); 
    loglog(imag(s), resp_soQuadBT_Hermite, '-.', 'linewidth', 2, 'color', ColMat(3,:)); 
    leg = legend('Full-order', 'soQuadBT', 'soQuadBT (Hermite)', ...
         'location', 'southeast', 'orientation', 'horizontal', 'interpreter', 'latex');
    leg = legend('Full-order', 'soQuadBT (Hermite)', 'location', 'southeast', ...
        'orientation', 'horizontal', 'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')
    
    % Relative errors.
    subplot(2,1,2)
    loglog(imag(s), relSVError_soQuadBT,         '-o', 'linewidth', 2, 'color', ColMat(2,:)); hold on
    loglog(imag(s), relSVError_soQuadBT_Hermite, '-.', 'linewidth', 2, 'color', ColMat(3,:));
    leg = legend('soQuadBT', 'soQuadBT (Hermite)', ...
         'location', 'southeast', 'orientation', 'horizontal', 'interpreter', 'latex');
    leg = legend('soQuadBT (Hermite)', 'location', 'southeast', 'orientation', ...
        'horizontal', 'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', ...
        fs, 'interpreter', 'latex')
end

% Store data.
write = true;
if write
    magMatrix = [imag(s)', GfoResp, resp_soQuadBT, resp_soQuadBT_Hermite];
    dlmwrite('results/butterfly_Hermite_r10_N200_1e4to1e6_mag.dat', magMatrix, ...
        'delimiter', '\t', 'precision', 8);
    errorMatrix = [imag(s)', relSVError_soQuadBT, relSVError_soQuadBT_Hermite];
    dlmwrite('results/butterfly_Hermite_r10_N200_1e4to1e6_error.dat', errorMatrix, ...
        'delimiter', '\t', 'precision', 8);
end

%% Stability checks.

rMax = 20;
for r = 2:2:rMax
    % Compute reduced-order model for given order r.

    % Reductor.
    [Z_soQuadBT, S_soQuadBT, Y_soQuadBT] = svd(Mbar_soQuadBT_Hermite);

    % Reduced model matrices.
    Mr_soQuadBT_Hermite  = eye(r, r);
    Kr_soQuadBT_Hermite  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Kbar_soQuadBT_Hermite*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Dr_soQuadBT_Hermite  = alpha*Mr_soQuadBT_Hermite + beta*Kr_soQuadBT_Hermite;
    Cpr_soQuadBT_Hermite = CpBar_soQuadBT_Hermite*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Br_soQuadBT_Hermite  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Bbar_soQuadBT_Hermite;

    Efor_Hermite = [eye(r, r), zeros(r, r); zeros(r, r), Mr_soQuadBT_Hermite];            % Descriptor matrix; Efo = [I, 0: 0, M]
    Afor_Hermite = [zeros(r, r), eye(r, r); -Kr_soQuadBT_Hermite, - Dr_soQuadBT_Hermite]; % Afo = [0, I; -K, -D]

    % Reductor.
    [Z_soQuadBT, S_soQuadBT, Y_soQuadBT] = svd(Mbar_soQuadBT);

    % Reduced model matrices.
    Mr_soQuadBT_singleOut  = eye(r, r);
    Kr_soQuadBT_singleOut  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Kbar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Dr_soQuadBT_singleOut  = alpha*Mr_soQuadBT_singleOut + beta*Kr_soQuadBT_singleOut;
    Cpr_soQuadBT_singleOut = CpBar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Br_soQuadBT_singleOut  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Bbar_soQuadBT;

    Efor = [eye(r, r), zeros(r, r); zeros(r, r), Mr_soQuadBT_singleOut];    % Descriptor matrix; Efo = [I, 0: 0, M]
    Afor = [zeros(r, r), eye(r, r); -Kr_soQuadBT_singleOut, - Dr_soQuadBT_singleOut]; % Afo = [0, I; -K, -D]

    % Compute all eigs to validate stability via linearized problem. 
    tmp1 = eig(Afor_Hermite, Efor_Hermite);
    if any(real(tmp1) > 0)
        fprintf(1, 'ORDER r = %d soQuadBT (Hermite) REDUCED MODEL IS UNSTABLE!\n', r)
    else
        fprintf(1, 'ORDER r = %d soQuadBT (Hermite) REDUCED MODEL IS STABLE!\n', r)
    end

    tmp2 = eig(Afor, Efor);
    if any(real(tmp2) > 0)
        fprintf(1, 'ORDER r = %d soQuadBT REDUCED MODEL IS UNSTABLE!\n', r)
    else
        fprintf(1, 'ORDER r = %d soQuadBT REDUCED MODEL IS STABLE!\n', r)
    end
    fprintf(1, '-----------------------------------------------\n')

end

%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off

