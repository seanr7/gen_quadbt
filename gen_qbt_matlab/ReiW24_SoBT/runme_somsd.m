%% RUNME_SOMSD
% Script file to run all experiments involving the mass-spring-damper
% system.
%

%
% This file is part of the archive Code, Data and Results for Numerical 
% Experiments in "Data-driven balanced truncation for second-order systems
% with generalized proportional damping"
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
fprintf(1, 'Loading mass-spring-damper problem.\n')
fprintf(1, '-----------------------------------\n');

% From: [Truhar and Veselic 2009, Ex. 2]
% Rayleigh Damping: D = alpha*M + beta*K
%   alpha = 2e-3;
%   beta  = 2e-3;
% load('data/MSDRayleigh_Cv.mat')
% alpha = 2e-3;
% beta  = alpha;

n1 = 300;   alpha = .002;   beta = alpha;   v = 0;

[M, D, K] = triplechain_MSD(n1, alpha, beta, v);

% Input, output, state dimensions.
n = size(full(K), 1);   p = 1;  m = 1;

% Input and velocity-output matrices.
B  = ones(n, m);   
Cv = ones(p, n);

%% Reduced order models.
% Test performance from i[1e-3, 1e1].
a = -3;  b = 1;  nNodes = 200;          

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
r = 20;

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
        GsRight(:, :, k) = (nodesRight(k).*Cv)*((nodesRight(k)^2.*M + nodesRight(k).*D + K)\B);
        GsLeft(:, :, k)  = (nodesLeft(k).*Cv)*((nodesLeft(k)^2.*M  + nodesLeft(k).*D  + K)\B);
        fprintf(1, 'Solves finished in %.2f s.\n',toc)
        fprintf(1, '-----------------------------\n');
    end
    save('results/MSD_Cv_samples_N200_1e-3to1e1.mat', 'GsLeft', 'GsRight', ...
        'nodesLeft', 'nodesRight')
else
    fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
    fprintf(1, '-------------------------------------------\n')
    load('results/MSD_Cv_samples_N200_1e-3to1e1.mat', 'GsLeft', 'GsRight')
end

% Non-intrusive methods.
%% 1. soQuadBT.
fprintf(1, 'BUILDING LOEWNER MATRICES (soQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;

% Loewner matrices.
[Mbar_soQuadBT, ~, Kbar_soQuadBT, Bbar_soQuadBT, CvBar_soQuadBT] = ...
    so_loewner_factory(nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeft, ...
                       GsRight, 'Rayleigh', [alpha, beta], 'Velocity');
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
Bbar_soQuadBT = Jp'*Bbar_soQuadBT;    CvBar_soQuadBT = CvBar_soQuadBT*Jm;
Bbar_soQuadBT = real(Bbar_soQuadBT);  CvBar_soQuadBT = real(CvBar_soQuadBT);
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
    Cvr_soQuadBT = CvBar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Br_soQuadBT  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Bbar_soQuadBT;
        
    fprintf(1, 'REDUCED-ORDER MODEL COMPUTED IN %.2f s\n', toc(timeRed))
    fprintf(1, '--------------------------------------\n')

    filename = 'results/roMSD_Cv_soQuadBT_r20_N200_1e-3to1e1.mat';
    save(filename, 'Mr_soQuadBT', 'Dr_soQuadBT', 'Kr_soQuadBT', 'Br_soQuadBT', 'Cvr_soQuadBT');
else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (soQuadBT).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roMSD_Cv_soQuadBT_r20_N200_1e-3to1e1.mat')
end

%% 2. soLoewner.
fprintf(1, 'BUILDING LOEWNER MATRICES (soLoewner).\n')
fprintf(1, '---------------------------------------\n')
timeLoewner = tic;

% Loewner matrices.
[Mbar_soLoewner, ~, Kbar_soLoewner, Bbar_soLoewner, CvBar_soLoewner] = ...
    so_loewner_factory(nodesLeft, nodesRight, ones(nNodes, 1), ones(nNodes, 1), GsLeft, ...
                       GsRight, 'Rayleigh', [alpha, beta], 'Velocity');
fprintf(1, 'CONSTRUCTION OF LOEWNER MATRICES FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

% Make real valued.
Mbar_soLoewner = Jp'*Mbar_soLoewner*Jm; Kbar_soLoewner  = Jp'*Kbar_soLoewner*Jm;   
Mbar_soLoewner = real(Mbar_soLoewner);  Kbar_soLoewner  = real(Kbar_soLoewner);  
Bbar_soLoewner = Jp'*Bbar_soLoewner;    CvBar_soLoewner = CvBar_soLoewner*Jm;
Bbar_soLoewner = real(Bbar_soLoewner);  CvBar_soLoewner = real(CvBar_soLoewner);
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
    Cvr_soLoewner = CvBar_soLoewner*Xr_soLoewner(:, 1:r);

    fprintf(1, 'REDUCED-ORDER MODEL COMPUTED IN %.2f s\n', toc(timeRed))
    fprintf(1, '--------------------------------------\n')
    
    filename = 'results/roMSD_Cv_soQuadBT_r20_N200_1e-1to1e3.mat';
    save(filename, 'Mr_soLoewner', 'Dr_soLoewner', 'Kr_soLoewner', 'Br_soLoewner', ...
        'Cvr_soLoewner');
else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (soLoewner).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roMSD_Cv_soQuadBT_r20_N200_1e-1to1e3.mat')
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
    % Reductor.
    [Z_foQuadBT, S_foQuadBT, Y_foQuadBT] = svd(Ebar_foQuadBT);

    % Reduced model matrices.
    Er_foQuadBT  = eye(r, r);
    Ar_foQuadBT  = (S_foQuadBT(1:r, 1:r)^(-1/2)*Z_foQuadBT(:, 1:r)')*Abar_foQuadBT*(Y_foQuadBT(:, 1:r)*S_foQuadBT(1:r, 1:r)^(-1/2));
    Cr_foQuadBT  = Cbar_foQuadBT*(Y_foQuadBT(:, 1:r)*S_foQuadBT(1:r, 1:r)^(-1/2));
    Br_foQuadBT  = (S_foQuadBT(1:r, 1:r)^(-1/2)*Z_foQuadBT(:, 1:r)')*Bbar_foQuadBT;
    
    filename = 'results/roMSD_Cv_foQuadBT_r20_N200_1e-1to1e3.mat';
    save(filename, 'Er_foQuadBT', 'Ar_foQuadBT', 'Br_foQuadBT', 'Cr_foQuadBT');

else
    load('results/roMSD_Cv_foQuadBT_r20_N200_1e-1to1e3.mat')
end

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
    soSys.Cp = zeros(p, n);
    soSys.Cv = Cv;
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
    Cvr_soBT = soBTRom.Cv;

    fprintf(1, 'REDUCED-ORDER MODEL COMPUTED IN %.2f s\n', toc(timeRed))
    fprintf(1, '--------------------------------------\n')
    
    filename = 'results/roMSD_Cv_soBT_r20.mat';
    save(filename, 'Mr_soBT', 'Dr_soBT', 'Kr_soBT', 'Br_soBT', 'Cvr_soBT');
else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (soBT).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roMSD_Cv_soBT_r20.mat')
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

    Cfo             = spalloc(1, 2*n, nnz(Cv)); % Cfo = [0, Cv];
    Cfo(:, n+1:2*n) = Cv; 

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
    
    filename = 'results/roMSD_Cv_foBT_r20.mat';
    save(filename, 'Er_foBT', 'Ar_foBT', 'Br_foBT', 'Cr_foBT');
else
    fprintf(1, 'NOT RE-COMPUTING; LOAD REDUCED-ORDER MODEL (foBT).\n')
    fprintf(1, '--------------------------------------\n')
    load('results/roMSD_Cv_foBT_r20.mat')
end

%% Plots.
numSamples      = 500;
s               = 1i*logspace(-3, 1, numSamples);

% Transfer function evaluations.
Gr_soQuadBT  = zeros(p, m, numSamples);
Gr_soLoewner = zeros(p, m, numSamples);
Gr_foQuadBT  = zeros(p, m, numSamples);
Gr_soBT      = zeros(p, m, numSamples);
Gr_foBT      = zeros(p, m, numSamples);

% Magnitude response and errors.
resp_soQuadBT      = zeros(numSamples, 1); % Response of (non-intrusive) soQuadBT reduced model
relError_soQuadBT  = zeros(numSamples, 1); % Rel. error due to (non-intrusive) soQuadBT reduced model
absError_soQuadBT  = zeros(numSamples, 1); % Abs. error due to (non-intrusive) soQuadBT reduced model

resp_soLoewner     = zeros(numSamples, 1); % Response of (non-intrusive) soLoewner reduced model
relError_soLoewner = zeros(numSamples, 1); % Rel. error due to (non-intrusive) soLoewner reduced model
absError_soLoewner = zeros(numSamples, 1); % Abs. error due to (non-intrusive) soLoewner reduced model

resp_foQuadBT      = zeros(numSamples, 1); % Response of (non-intrusive) foQuadBT reduced model
relError_foQuadBT  = zeros(numSamples, 1); % Rel. error due to (non-intrusive) foQuadBT reduced model
absError_foQuadBT  = zeros(numSamples, 1); % Abs. error due to (non-intrusive) foQuadBT reduced model

resp_soBT          = zeros(numSamples, 1); % Response of (intrusive) soBT reduced model
relError_soBT      = zeros(numSamples, 1); % Rel. error due to (intrusive) soBT reduced model
absError_soBT      = zeros(numSamples, 1); % Abs. error due to (intrusive) soBT reduced model

resp_foBT          = zeros(numSamples, 1); % Response of (intrusive) foBT reduced model
relError_foBT      = zeros(numSamples, 1); % Rel. error due to (intrusive) foBT reduced model
absError_foBT      = zeros(numSamples, 1); % Abs. error due to (intrusive) foBT reduced model

% Full-order simulation data.
recompute = false;
if recompute
    Gfo     = zeros(p, m, numSamples);
    GfoResp = zeros(numSamples, 1);
    GfoFrob = zeros(numSamples, 1);
    fprintf(1, 'Sampling full-order transfer function along i[1e-3, 1e1].\n')
    fprintf(1, '--------------------------------------------------------\n')
    for ii = 1:numSamples
        timeSolve = tic;
        fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, imag(s(ii)))
        Gfo(:, :, ii) = (s(ii)*Cv)*((s(ii)^2*M +s(ii)*D + K)\B);
        GfoResp(ii)   = abs(Gfo(:, :, ii));
        fprintf(1, 'k = %d solve finished in %.2f s\n', ii, toc(timeSolve))
    end
    save('results/MSD_Cv_samples_N500_1e-3to1e1.mat', 'Gfo', 'GfoResp')
else
    fprintf(1, 'Loading precomputed values.\n')
    fprintf(1, '--------------------------------------------------------\n')
    load('results/MSD_Cv_samples_N500_1e-3to1e1.mat')
end

% Compute frequency response along imaginary axis.
for ii=1:numSamples
    % Transfer functions.
    Gr_soQuadBT(:, :, ii)  = (s(ii)*Cvr_soQuadBT)*((s(ii)^2*Mr_soQuadBT + s(ii)*Dr_soQuadBT + Kr_soQuadBT)\Br_soQuadBT);
    Gr_soLoewner(:, :, ii) = (s(ii)*Cvr_soLoewner)*((s(ii)^2*Mr_soLoewner + s(ii)*Dr_soLoewner + Kr_soLoewner)\Br_soLoewner);
    Gr_foQuadBT(:, :, ii)  = Cr_foQuadBT*((s(ii)*Er_foQuadBT - Ar_foQuadBT)\Br_foQuadBT);
    Gr_soBT(:, :, ii)      = (s(ii)*Cvr_soBT)*((s(ii)^2*Mr_soBT + s(ii)*Dr_soBT + Kr_soBT)\Br_soBT);
    Gr_foBT(:, :, ii)      = Cr_foBT*((s(ii)*Er_foBT - Ar_foBT)\Br_foBT);

    % Response and errors. 
    resp_soQuadBT(ii)      = abs(Gr_soQuadBT(:, :, ii)); 
    absError_soQuadBT(ii)  = abs(Gfo(:, :, ii) - Gr_soQuadBT(:, :, ii));
    relError_soQuadBT(ii)  = absError_soQuadBT(ii)/GfoResp(ii);

    resp_foQuadBT(ii)      = abs(Gr_foQuadBT(:, :, ii)); 
    absError_foQuadBT(ii)  = abs(Gfo(:, :, ii) - Gr_foQuadBT(:, :, ii)); 
    relError_foQuadBT(ii)  = absError_foQuadBT(ii)/GfoResp(ii); 

    resp_soLoewner(ii)     = abs(Gr_soLoewner(:, :, ii)); 
    absError_soLoewner(ii) = abs(Gfo(:, :, ii) - Gr_soLoewner(:, :, ii));
    relError_soLoewner(ii) = absError_soLoewner(ii)/GfoResp(ii); 

    resp_soBT(ii)          = abs(Gr_soBT(:, :, ii)); 
    absError_soBT(ii)      = abs(Gfo(:, :, ii) - Gr_soBT(:, :, ii));
    relError_soBT(ii)      = absError_soBT(ii)/GfoResp(ii); 

    resp_foBT(ii)          = abs(Gr_foBT(:, :, ii)); 
    absError_foBT(ii)      = abs(Gfo(:, :, ii) - Gr_foBT(:, :, ii));
    relError_foBT(ii)      = absError_foBT(ii)/GfoResp(ii); 
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
    loglog(imag(s), GfoResp,        '-o', 'linewidth', 2, 'color', ColMat(1,:)); hold on
    loglog(imag(s), resp_soQuadBT,  '--', 'linewidth', 2, 'color', ColMat(2,:)); 
    loglog(imag(s), resp_soLoewner, '--', 'linewidth', 2, 'color', ColMat(3,:)); 
    loglog(imag(s), resp_foQuadBT,  '-.', 'linewidth', 2, 'color', ColMat(4,:)); 
    loglog(imag(s), resp_soBT,      '-.', 'linewidth', 2, 'color', ColMat(5,:)); 
    loglog(imag(s), resp_foBT,      '-.', 'linewidth', 2, 'color', ColMat(6,:)); 
    leg = legend('Full-order', 'soQuadBT', 'soLoewner', 'foQuadBT', 'soBT', 'foBT', ...
        'location', 'southeast', 'orientation', 'horizontal', 'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')
    
    % Relative errors
    subplot(2,1,2)
    loglog(imag(s), relError_soQuadBT,  '-o', 'linewidth', 2, 'color', ColMat(2,:)); hold on
    loglog(imag(s), relError_soLoewner, '-*', 'linewidth', 2, 'color', ColMat(3,:));
    loglog(imag(s), relError_foQuadBT,  '-*', 'linewidth', 2, 'color', ColMat(4,:));
    loglog(imag(s), relError_soBT,      '-*', 'linewidth', 2, 'color', ColMat(5,:));
    loglog(imag(s), relError_foBT,      '-*', 'linewidth', 2, 'color', ColMat(6,:));
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
    dlmwrite('results/MSD_Cv_r20_N200_1e-3to1e1_mag.dat', magMatrix, ...
        'delimiter', '\t', 'precision', 8);
    errorMatrix = [imag(s)', relError_soQuadBT, relError_soLoewner, relError_foQuadBT, ...
        relError_soBT, relError_foBT];
    dlmwrite('results/MSD_Cv_r20_N200_1e-3to1e1_error.dat', errorMatrix, ...
        'delimiter', '\t', 'precision', 8);
end

%% Error measures.
% Print errors.
fprintf(1, 'Order r = %d.\n', r)
fprintf(1, '--------------\n')
fprintf(1, 'Relative H-infty error due to soQuadBT : %.16f \n', max((absError_soQuadBT))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to soLoewner: %.16f \n', max((absError_soLoewner))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to foQuadBT : %.16f \n', max((absError_foQuadBT))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to soBT     : %.16f \n', max((absError_soBT))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to foBT     : %.16f \n', max((absError_foBT))./max((GfoResp)))
fprintf(1, '------------------------------------------------------------\n')
fprintf(1, 'Relative H-2 error due to soQuadBT     : %.16f \n', sqrt(sum(absError_soQuadBT.^2)/sum(GfoResp.^2)))
fprintf(1, 'Relative H-2 error due to soLoewner    : %.16f \n', sqrt(sum(absError_soLoewner.^2)/sum(GfoResp.^2)))
fprintf(1, 'Relative H-2 error due to foQuadBT     : %.16f \n', sqrt(sum(absError_foQuadBT.^2)/sum(GfoResp.^2)))
fprintf(1, 'Relative H-2 error due to soBT         : %.16f \n', sqrt(sum(absError_soBT.^2)/sum(GfoResp.^2)))
fprintf(1, 'Relative H-2 error due to foBT         : %.16f \n', sqrt(sum(absError_foBT.^2)/sum(GfoResp.^2)))
fprintf(1, '------------------------------------------------------------\n')


%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off
