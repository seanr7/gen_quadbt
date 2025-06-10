%% RUNME_SOPLATE
% Script file to run all experiments involving the model of a plate with 
% tuned vibration absorbers (TVAs).
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
fprintf(1, 'Loading plate with TVAs.\n')
fprintf(1, '----------------------------------------------\n')

% From: https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Plate_with_tuned_vibration_absorbers
% Hysteretic damping.
%   eta = .001.
load('data/soplateTVA.mat')
Cp = C;

% Hysteretic damping coefficient.
eta = .001;

%% Reduced order models.
% Test performance from 0 to 250 Hz.
% Frequencies used in the simulation.
%   s    = 1i*linspace(0, 2*pi*250, 250); 
%   s_hz = imag(s)/2/pi; 
nNodes = 250;          

% Prepare quadrature weights and nodes according to Trapezoidal rule.
omega      = 1i*(linspace(0, 2*pi*250, nNodes)');
nodesLeft  = omega(1:2:end);    
nodesRight = omega(2:2:end); 

% Close left and right nodes under complex conjugation.
nodesLeft  = ([nodesLeft; conj(flipud(nodesLeft))]);     
nodesRight = ([nodesRight; conj(flipud(nodesRight))]);

% Weights.
weightsRight = [nodesRight(2) - nodesRight(1); nodesRight(3:end) - nodesRight(1:end-2); ...
    nodesRight(end) - nodesRight(end-1)]./2;
weightsRight = sqrt(1/(2*pi))*sqrt(abs(weightsRight));   
weightsLeft  = [nodesLeft(2) - nodesLeft(1); nodesLeft(3:end) - nodesLeft(1:end-2); ...
    nodesLeft(end) - nodesLeft(end-1)]./2; 
weightsLeft  = sqrt(1/(2*pi))*sqrt(abs(weightsLeft)); 

% Order of reduction.
r = 75;

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
        % Note: No s*D due to Hysteretic damping.
        GsRight(:, :, k) = Cp*((nodesRight(k)^2.*M + D + K)\B);
        GsLeft(:, :, k)  = Cp*((nodesLeft(k)^2.*M  + D + K)\B);
        fprintf(1, 'Solves finished in %.2f s.\n',toc)
        fprintf(1, '-----------------------------\n');
    end
    save('results/plateTVA_samples_N250_0to250Hz.mat', 'GsLeft', 'GsRight', ...
        'nodesLeft', 'nodesRight')
else
    fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
    fprintf(1, '-------------------------------------------\n')
    load('results/plateTVA_samples_N250_0to250Hz.mat', ...
        'GsLeft', 'GsRight')
end

% Non-intrusive methods.
%% 1. soQuadBT.
fprintf(1, 'BUILDING LOEWNER QUADRUPLE (soQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;

% Loewner matrices.
[Mbar_soQuadBT, Dbar_soQuadBT, Kbar_soQuadBT, Bbar_soQuadBT, CpBar_soQuadBT] = ...
    so_loewner_factory(nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeft, ...
                       GsRight, 'Structural', eta, 'Position');
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

recomputeModel = true;
if recomputeModel
    % Reductor.
    [Z_soQuadBT, S_soQuadBT, Y_soQuadBT] = svd(Mbar_soQuadBT);

    % Reduced model matrices.
    Mr_soQuadBT  = eye(r, r);
    Kr_soQuadBT  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Kbar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Dr_soQuadBT  = 1i*eta*Kr_soQuadBT;
    Cpr_soQuadBT = CpBar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
    Br_soQuadBT  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Bbar_soQuadBT;
    
    filename = 'results/roPlateTVA_soQuadBT_r75_N250_0to250Hz.mat';
    save(filename, 'Mr_soQuadBT', 'Dr_soQuadBT', 'Kr_soQuadBT', 'Br_soQuadBT', 'Cpr_soQuadBT');
else
    load('results/roPlateTVA_soQuadBT_r75_N250_0to250Hz.mat')
end

%% 2. soLoewner.
fprintf(1, 'BUILDING LOEWNER QUADRUPLE (soLoewner).\n')
fprintf(1, '---------------------------------------\n')
timeLoewner = tic;
[Mbar_soLoewner, Dbar_soLoewner, Kbar_soLoewner, Bbar_soLoewner, CpBar_soLoewner] = ...
    so_loewner_factory(nodesLeft, nodesRight, ones(nNodes, 1), ones(nNodes, 1), GsLeft, ...
                       GsRight, 'Structural', eta, 'Position');
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

recomputeModel = true;
if recomputeModel
    % Reductor.
    % Relevant SVDs.
    [Yl_soLoewner, Sl_soLoewner, ~] = svd([-Mbar_soLoewner, Kbar_soLoewner], 'econ');
    [~, Sr_soLoewner, Xr_soLoewner] = svd([-Mbar_soLoewner; Kbar_soLoewner], 'econ');
    
    % Compress.
    Mr_soLoewner  = Yl_soLoewner(:, 1:r)'*Mbar_soLoewner*Xr_soLoewner(:, 1:r); % This needs a -?
    Kr_soLoewner  = Yl_soLoewner(:, 1:r)'*Kbar_soLoewner*Xr_soLoewner(:, 1:r);
    Dr_soLoewner  = 1i*eta*Kr_soLoewner;
    Br_soLoewner  = Yl_soLoewner(:, 1:r)'*Bbar_soLoewner;
    Cpr_soLoewner = CpBar_soLoewner*Xr_soLoewner(:, 1:r);
    
    filename = 'results/roPlateTVA_soLoewner_r75_N250_0to250Hz.mat';
    save(filename, 'Mr_soLoewner', 'Dr_soLoewner', 'Kr_soLoewner', 'Br_soLoewner', 'Cpr_soLoewner');
else
    load('results/roPlateTVA_soLoewner_r75_N250_0to250Hz.mat')
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

recomputeModel = true;
if recomputeModel
    % Reductor.
    [Z_foQuadBT, S_foQuadBT, Y_foQuadBT] = svd(Ebar_foQuadBT);

    % Reduced model matrices.
    Er_foQuadBT  = eye(r, r);
    Ar_foQuadBT  = (S_foQuadBT(1:r, 1:r)^(-1/2)*Z_foQuadBT(:, 1:r)')*Abar_foQuadBT*(Y_foQuadBT(:, 1:r)*S_foQuadBT(1:r, 1:r)^(-1/2));
    Cr_foQuadBT  = Cbar_foQuadBT*(Y_foQuadBT(:, 1:r)*S_foQuadBT(1:r, 1:r)^(-1/2));
    Br_foQuadBT  = (S_foQuadBT(1:r, 1:r)^(-1/2)*Z_foQuadBT(:, 1:r)')*Bbar_foQuadBT;
    
    filename = 'results/roPlateTVA_foQuadBT_r75_N250_0to250Hz.mat';
    save(filename, 'Er_foQuadBT', 'Ar_foQuadBT', 'Br_foQuadBT', 'Cr_foQuadBT');
else
    load('results/roPlateTVA_foQuadBT_r75_N250_0to250Hz.mat')
end

%% Plots.
numSamples      = 500;
s               = 1i*linspace(0, 2*pi*250, numSamples);
s_hz            = imag(s)/2/pi;

% Transfer function evaluations.
Gr_soQuadBT  = zeros(p, m, numSamples);
Gr_soLoewner = zeros(p, m, numSamples);
Gr_foQuadBT  = zeros(p, m, numSamples);

% Magnitude response and errors.
resp_soQuadBT          = zeros(numSamples, 1); % Response of (non-intrusive) soQuadBT reduced model
relError_soQuadBT    = zeros(numSamples, 1); % Rel. error due to (non-intrusive) soQuadBT reduced model
absError_soQuadBT    = zeros(numSamples, 1); % Abs. error due to (non-intrusive) soQuadBT reduced model

resp_soLoewner         = zeros(numSamples, 1); % Response of (non-intrusive) soLoewner reduced model
relError_soLoewner   = zeros(numSamples, 1); % Rel. error due to (non-intrusive) soLoewner reduced model
absError_soLoewner   = zeros(numSamples, 1); % Abs. error due to (non-intrusive) soLoewner reduced model

resp_foQuadBT          = zeros(numSamples, 1); % Response of (non-intrusive) foQuadBT reduced model
relError_foQuadBT    = zeros(numSamples, 1); % Rel. error due to (non-intrusive) foQuadBT reduced model
absError_foQuadBT    = zeros(numSamples, 1); % Abs. error due to (non-intrusive) foQuadBT reduced model

% Full-order simulation data.
recompute = false;
if recompute
    Gfo     = zeros(p, m, numSamples);
    GfoResp = zeros(numSamples, 1);
    GfoFrob = zeros(numSamples, 1);
    fprintf(1, 'Sampling full-order transfer function along 0 to 250 Hz.\n')
    fprintf(1, '--------------------------------------------------------\n')
    for ii = 1:numSamples
        timeSolve = tic;
        fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, imag(s(ii)))
        % No s(ii)*D because Hysteretic.
        Gfo(:, :, ii) = Cp*((s(ii)^2.*M + D + K)\B);
        GfoResp(ii)   = abs(Gfo(:, :, ii));  
        fprintf(1, 'k = %d solve finished in %.2f s\n', ii, toc(timeSolve))
    end
    save('results/_samples_N500_0to250Hz.mat', 'Gfo', 'GfoResp')
else
    fprintf(1, 'Loading precomputed values.\n')
    fprintf(1, '--------------------------------------------------------\n')
    load('results/_samples_N500_0to250Hz.mat')
end

% Compute frequency response along imaginary axis.
for ii=1:numSamples
    % Transfer functions.
    Gr_soQuadBT(:, :, ii)  = Cpr_soQuadBT*((s(ii)^2.*Mr_soQuadBT + Dr_soQuadBT + Kr_soQuadBT)\Br_soQuadBT);
    Gr_soLoewner(:, :, ii) = Cpr_soLoewner*((s(ii)^2.*Mr_soLoewner + Dr_soLoewner + Kr_soLoewner)\Br_soLoewner);
    Gr_foQuadBT(:, :, ii)  = Cr_foQuadBT*((s(ii).*Er_foQuadBT - Ar_foQuadBT)\Br_foQuadBT);

    % Response and errors. 
    resp_soQuadBT(ii)      = abs(Gr_soQuadBT(:, :, ii)); 
    absError_soQuadBT(ii)  = abs(Gfo(:, :, ii) - Gr_soQuadBT(:, :, ii));
    relError_soQuadBT(ii)  = absError_soQuadBT(ii)/GfoResp(ii);

    resp_soLoewner(ii)     = abs(Gr_soLoewner(:, :, ii)); 
    absError_soLoewner(ii) = abs(Gfo(:, :, ii) - Gr_soLoewner(:, :, ii));
    relError_soLoewner(ii) = absError_soLoewner(ii)/GfoResp(ii);

    resp_foQuadBT(ii)      = abs(Gr_foQuadBT(:, :, ii)); 
    absError_foQuadBT(ii)  = abs(Gfo(:, :, ii) - Gr_foQuadBT(:, :, ii)); 
    relError_foQuadBT(ii)  = absError_foQuadBT(ii)/GfoResp(ii);
end

% Convert to [dB].
GfoResp            = 10*log10(abs(GfoResp)/1e-9); 
resp_soQuadBT      = 10*log10(abs(resp_soQuadBT)/1e-9); 
resp_soLoewner     = 10*log10(abs(resp_soLoewner)/1e-9); 
resp_foQuadBT      = 10*log10(abs(resp_foQuadBT)/1e-9); 

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
    plot((s_hz), GfoResp,        '-o', 'linewidth', 2, 'color', ColMat(1,:)); hold on
    plot((s_hz), resp_soQuadBT,  '--', 'linewidth', 2, 'color', ColMat(2,:)); 
    plot((s_hz), resp_soLoewner, '--', 'linewidth', 2, 'color', ColMat(3,:));  
    plot((s_hz), resp_foQuadBT,  '-.', 'linewidth', 2, 'color', ColMat(4,:)); 
    leg = legend('Full-order', 'soQuadBT', 'soLoewner', 'foQuadBT', 'location', 'southeast', 'orientation', 'horizontal', ...
        'interpreter', 'latex');
    xlim([(s_hz(1)), (s_hz(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('s [Hz]', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('Magnitude', 'fontsize', fs, 'interpreter', 'latex')
    
    % Relative errors
    subplot(2,1,2)
    semilogy((s_hz), relError_soQuadBT,  '-o', 'linewidth', 2, 'color', ColMat(2,:)); hold on
    semilogy((s_hz), relError_soLoewner, '-*', 'linewidth', 2, 'color', ColMat(3,:));
    semilogy((s_hz), relError_foQuadBT,  '-*', 'linewidth', 2, 'color', ColMat(4,:));
    leg = legend('soQuadBT', 'soLoewner', 'foQuadBT', 'location', 'southeast', 'orientation', 'horizontal', ...
        'interpreter', 'latex');
    xlim([(s_hz(1)), (s_hz(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('s [Hz]', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('Relative error', 'fontsize', ...
        fs, 'interpreter', 'latex')
end

% Store data.
write = true;
if write
    magMatrix = [(s_hz)', GfoResp, resp_soQuadBT, resp_soLoewner, resp_foQuadBT];
    dlmwrite('results/plateTVA_r75_N250_0to250Hz_mag.dat', magMatrix, ...
        'delimiter', '\t', 'precision', 8);
    errorMatrix = [(s_hz)', relError_soQuadBT, relError_soLoewner, relError_foQuadBT];
    dlmwrite('results/plateTVA_r75_N250_0to250Hz_error.dat', errorMatrix, ...
        'delimiter', '\t', 'precision', 8);
end


%% Error measures.
% Print errors.
fprintf(1, 'Order r = %d.\n', r)
fprintf(1, '--------------\n')
fprintf(1, 'Relative H-infty error due to soQuadBT : %.16f \n', max((absError_soQuadBT))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to soLoewner: %.16f \n', max((absError_soLoewner))./max((GfoResp)))
fprintf(1, 'Relative H-infty error due to foQuadBT : %.16f \n', max((absError_foQuadBT))./max((GfoResp)))
fprintf(1, '------------------------------------------------------------\n')
fprintf(1, 'Relative H-2 error due to soQuadBT     : %.16f \n', sqrt(sum(absError_soQuadBT.^2)/sum(GfoResp.^2)))
fprintf(1, 'Relative H-2 error due to soLoewner    : %.16f \n', sqrt(sum(absError_soLoewner.^2)/sum(GfoResp.^2)))
fprintf(1, 'Relative H-2 error due to foQuadBT     : %.16f \n', sqrt(sum(absError_foQuadBT.^2)/sum(GfoResp.^2)))
fprintf(1, '------------------------------------------------------------\n')

%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off
