%% RUNME_MSD
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
fprintf(1, 'Loading toy mass spring damper model.\n')
fprintf(1, '-------------------------------------\n');

write = 1;
if write
    % Proportional damping D = alpha*M + beta*K
    n1 = 100;   alpha = .002;   beta = alpha;   v = 5;
    [M, D, K] = triplechain_MSD(n1, alpha, beta, v);

    % Input, output, state dimensions
    n = size(full(K), 1);   p = 1;  m = 1;
    
    % Input and output matrices
    B  = ones(n, m);    C  = ones(p, n);
    save('data/MSD.mat', 'M', 'K', 'C', 'B', 'n', 'm', 'p')
else
    load('data/MSD.mat')
end


%% Compute intermediate reduced model.
% Project down full-order model to intermediate reduced model for the
% purpose of computing frequency-limited Gramians.

% Option to recompute or load intermediate reduced model.
% recompute = false;
recompute = true;
if recompute
    % Project down using r < noInterpPoints interpolation points along iR
    noInterpPoints = 500;                                % Number of frequencies to sample at
    interpPoints   = 1i*logspace(0, 6, noInterpPoints); % Interpolation points

    % Compute Galerkin model reduction basis
    linear_solves = tic;
    fprintf(1, 'Computing Galerkin model reduction basis and intermediate reduced model.\n')
    fprintf(1, '------------------------------------------------------------------------\n')
    % Space allocation for basis and transfer function values
    V       = zeros(n, m*noInterpPoints);
    Gfo     = zeros(p, m, noInterpPoints);
    res_fom = zeros(noInterpPoints, 1);
    for k = 1:noInterpPoints
        this_solve = tic;
        fprintf(1, 'CURRENT ITERATE IS k = %d\n', k)
        fprintf(1, '---------------------------\n')
        V(:, k) = (interpPoints(k)^2.*M + interpPoints(k).*D + K)\B;
        fprintf(1, 'k = %d SOLVE FINISHED IN %.2f s\n', k, toc(this_solve))
        fprintf(1, '--------------------------------\n')

        % While we're at it, evaluate transfer function
        Gfo(:, :, k) = C*V(:, k); 
        res_fom(k)   = norm(Gfo(:, :, k), 2);
    end
    fprintf(1, 'GALERKIN BASIS COMPUTED IN %.2f s\n', toc(linear_solves))
    fprintf(1, '-----------------------------------\n')

    % Orthogonalize model reduction basis to avoid ill-conditioning
    orth = tic;
    fprintf(1, 'ORTHOGONALIZING PROJECTION MATRIX.\n')
    fprintf(1, '----------------------------------\n')
    VExpanded     = [real(V), imag(V)];              % To ensure intermediate model is real-valued
    tol           = 10e-14;                          % Singular value cutoff
    [U, Sigma, ~] = svd(VExpanded);
    r             = find(diag(Sigma)>tol, 1,'last'); % Finds index of last singular value > tol
    Vorth         = U(:, 1:r);  
    fprintf(1, 'PROJECTION MATRIX MATRIX ORTHOGONALIZED IN %.2f s\n', toc(orth))
    fprintf(1, '---------------------------------------------------\n')

    % Project down full-order matrices
    proj = tic;
    fprintf(1, 'COMPUTING INTERMEDIATE REDUCED MODEL VIA GALERKIN PROJECTION.\n')
    fprintf(1, '-------------------------------------------------------------\n')
    Mr = Vorth.'*M*Vorth;   Dr = Vorth.'*D*Vorth;   Kr = Vorth.'*K*Vorth;   
    Br = Vorth.'*B;         Cr = C*Vorth;

    % Now, convert to first-order realization (fl Gramian solver
    % implemented for first-order systems)
    Er_fo_int                   = zeros(2*r, 2*r); % Descriptor matrix; Efo = [I, 0: 0, M]
    Er_fo_int(1:r, 1:r)         = eye(r, r);       % (1, 1) block
    Er_fo_int(r+1:2*r, r+1:2*r) = Mr;              % (2, 2) block is  mass matrix

    Ar_fo_int                   = zeros(2*r, 2*r); % Afo = [0, I; -K, -D]
    Ar_fo_int(1:r, r+1:2*r)     = eye(r, r);       % (1, 2) block of Afo
    Ar_fo_int(r+1:2*r, 1:r)     = -Kr;             % (2, 1) block is -K
    Ar_fo_int(r+1:2*r, r+1:2*r) = -Dr;             % (2, 2) block is -D

    Br_fo_int             = zeros(2*r, m); % Bfo = [0; B];
    Br_fo_int(r+1:2*r, :) = Br; 

    % Position input, only
    Cr_fo_int         = zeros(p, 2*r); % Cfo = [C, 0];
    Cr_fo_int(:, 1:r) = Cr; 

    % Save Gakerkin model reduction basis
    % save('data/ButterflyFOIntermedROM.mat', 'Vorth', 'Er_fo_int', 'Ar_fo_int', ...
    %     'Br_fo_int', 'Cr_fo_int')

    % Save full order simulation data
    % save('data/ButterflyFOSimulation.mat', 'Gfo', 'res_fom', 'interpPoints')
else
    fprintf(1, 'Not recomputing; load intermediate reduced model.\n')
    fprintf(1, '--------------------------------------------------\n')
    % Load intermediate reduced model 
    noInterpPoints = 500; 
    load('data/ButterflyFOIntermedROM.mat', 'Er_fo_int', 'Ar_fo_int', 'Br_fo_int', ...
        'Cr_fo_int')
    % Load full-order simulation data for later comparison
    load('data/ButterflyFOSimulation.mat', 'Gfo', 'res_fom', 'interpPoints')
end

% Validate intermediate reduced model.
% validate = false;
validate = true;
if validate
    % Load full-order simulation data
    % load('data/ButterflyFOSimulation.mat', 'Gfo', 'res_fom', 'interpPoints')
    % Simulate intermediate reduced model.
    Gro     = zeros(p, m, noInterpPoints);
    res_rom = zeros(noInterpPoints, 1);
    for k = 1:noInterpPoints
        fprintf(1, 'Frequency step %d, f=%.2f Hz ...\n ',k, imag(interpPoints(k))/(2*pi))
        Gro(:, :, k) = Cr_fo_int*((interpPoints(k).*Er_fo_int - Ar_fo_int)\Br_fo_int);
        res_rom(k)   = norm(Gro(:, :, k), 2);
        % Error at kth frequency
        fprintf(1, 'Relative error at frequency step %d, f=%.2f Hz: %.12f\n ',k, imag(interpPoints(k))/(2*pi), ...
            norm(Gfo(:, :, k) - Gro(:, :, k))/abs(res_fom(k)))
        fprintf(1, '---------------------------------------s-----------------------------\n')
    end
    % Plot colors
    ColMat = zeros(2,3);
    ColMat(1,:) = [0.8500    0.3250    0.0980];
    ColMat(2,:) = [0.3010    0.7450    0.9330];
    
    figure(1)
    % Pointwise L_infty error
    subplot(2,1,1)
    loglog(imag(interpPoints), abs(res_fom-res_rom)./abs(res_fom), '-.','color',ColMat(2,:),LineWidth=1);
    xlabel('Frequency [Hz]')
    ylabel('Error')
    xlim([imag(interpPoints(1)), imag(interpPoints(end))])
    
    % Magnitude
    subplot(2,1,2)
    loglog(imag(interpPoints), res_fom,'--o','color',ColMat(1,:),'markersize',4,LineWidth=1); 
    hold on;
    loglog(imag(interpPoints), res_rom, '-.','color',ColMat(2,:),LineWidth=1);
    xlabel('Frequency [Hz]')
    ylabel('Magnitude')
    legend('Full-order', 'Intermediate model')
    xlim([imag(interpPoints(1)), imag(interpPoints(end))])
end

%% Frequency-limited balanced truncation from data.
fprintf(1, '1a. Instantiate flQuadBTSampler class.\n')
fprintf(1, '--------------------------------------\n')

% In `frequency-limted Quadrature-based Balanced Truncation' (flQuadBT),
% the fl aspect of the approximation is captured by the range of
% frequencies in which the transfer function is sampled.

% Specify frequency-limited band along iR
a     = -2;
b     = -1;
Omega = [10^a, 10^b];
N     = 200;          % Number of quadrature nodes

% Prep fl quadrature weights and nodes
[nodesl, weightsl, nodesr, weightsr] = trap_rule([a, b], N, true);

% Sepcialized `flQuadBTSampler' class is used, to take advantage of the
% underlying sparse, second-order, system structure.
sampler = flQuadBTSampler_so(M, D, K, B, C, n, m, p, 0);

fprintf(1, '1b. Instantiate flQuadBTReductor class.\n')
fprintf(1, '------------------------------------------------\n')

flQuadBT_Engine = flQuadBTReductor(sampler, nodesl, nodesr, weightsl, ...
    weightsr);

fprintf(1, '1c. COMPUTING DATA AND LOEWNER QUADRUPLE.\n')
fprintf(1, '-----------------------------------------\n')
Loewner_start = tic;

% Loewner quadruple
Lbar = flQuadBT_Engine.Lbar;    Mbar = flQuadBT_Engine.Mbar; 
Hbar = flQuadBT_Engine.Hbar;    Gbar = flQuadBT_Engine.Gbar;

fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(Loewner_start))
fprintf(1, '------------------------------------------------------\n')

check_Loewner = 1;
if check_Loewner
    fprintf(1, 'Sanity check: Verify that the build of the Loewner matrices is correct.\n')
    fprintf(1, '-----------------------------------------------------------------------\n')

    % Compute quadrature-based factors for comparison
    factors_start = tic;
    fprintf(1, 'COMPUTING QUADRATURE-BASED FACTORS.\n')
    fprintf(1, '-----------------------------------\n')
    Uquad = sampler.quad_right_factor(nodesr, weightsr); % Approximately factors band-limited reachability Gramian
    Lquad = sampler.quad_left_factor(nodesl, weightsl);  % Approximately factors band-limited observability Gramian
    fprintf(1, 'QUADRATURE-BASED FACTORS COMPUTED IN %.2f s\n', toc(factors_start))
    fprintf(1, '-----------------------------------------\n')

    % Use class method to get first-order realization of second-order 
    % system for error checks
    [Efo, Afo, Bfo, Cfo, Dfo] = sampler.build_fo_realization; 

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Lbar: Error || Lbar - Lquad.H * Efo * Uquad ||_2: %.16f\n', norm(Lquad' * Efo * Uquad - Lbar, 2))
    fprintf('Check for Mbar: Error || Mbar - Lquad.H * Afo * Uquad ||_2: %.16f\n', norm(Lquad' * Afo * Uquad - Mbar, 2))
    fprintf('Check for Hbar: Error || Hbar - Lquad.H * Bfo         ||_2: %.16f\n', norm(Lquad' * Bfo - Hbar, 2))
    fprintf('Check for Gbar: Error || Gbar - Cfo * Uquad           ||_2: %.16f\n', norm(Cfo * Uquad - Gbar, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

%% Compute frequency-limited reduced models.
fprintf(1, '2. Computing frequency-limited reduced models.\n')
fprintf(1, '----------------------------------------------\n')

r = 25; % Reduction order

flQuadBT_start = tic;
fprintf(1, 'COMPUTING REDUCED MODEL VIA flQuadBT (non-intrusive).\n')
fprintf(1, '-----------------------------------------------------\n')

[Ar_flQuadBT, Br_flQuadBT, Cr_flQuadBT] = flQuadBT_Engine.reduce(r);
Dr_flQuadBT                             = 0;
Er_flQuadBT                             = eye(r, r);

fprintf(1, 'flQuadBT REDUCED MODEL COMPUTED IN %.2f s\n', toc(flQuadBT_start))
fprintf(1, '-----------------------------------------\n')

%
fprintf(1, 'COMPUTING REDUCED MODEL VIA flBT (intrusive).\n')
fprintf(1, '---------------------------------------------\n')
flQuadBT_start = tic;

% For stability of solver; use strictly dissipative companion form
% Compute realization alpha.
mfun  = @(x) (M * x + 0.25 * (D * (K \ (D * x))));
n     = size(M, 1);
alpha = 1 / (2 * eigs(mfun, n, D, 1, 'LR', struct('disp', 0)));


% Get strictly dissipative realization.
Afo = [-alpha * K, K - alpha * D; -K, -D + alpha * M];
Efo = [K, alpha * M; alpha * M, M];
Bfo = [alpha * B; B];
Cfo = [C, zeros(size(C))];

solve_opts = struct( ...
    'FctUpdate' , 1, ...
    'Freqs'     , Omega, ...
    'Info'      , 2, ...
    'MaxIter'   , 100, ...
    'MinIter'   , 10, ...
    'ModGramian', 0, ...
    'Npts'      , 1601, ...
    'Shifts'    , 'imaginary', ...
    'Solver'    , 'logm', ...
    'SolverFreq', 1, ...
    'StoreFacE' , 0, ...
    'StoreV'    , 0, ...
    'TolComp'   , eps, ...
    'TolLyap'   , 1.0e-12, ...
    'TolRHS'    , 1.0e-14, ...
    'TrueRes'   , 1, ...
    'L'         , [], ...
    'pL'        , []);

P_start = tic;
fprintf(1, 'SOLVING FOR FREQUENCY-LIMITED CONTROLLABILITY GRAMIAN.\n')
fprintf(1, '------------------------------------------------------\n')
[ZCont, YCont, infoCont] = freq_lyap_rksm(Afo, Bfo, Efo, solve_opts);
fprintf(1, 'COMPUTED IN %.2f s\n', toc(P_start))
[XCont, LCont] = eig(YCont);

Q_start = tic;
fprintf(1, 'SOLVING FOR FREQUENCY-LIMITED OBSERVABILITY GRAMIAN.\n')
fprintf(1, '----------------------------------------------------\n')
[ZObsv, YObsv, infoObsv] = freq_lyap_rksm(Afo', Cfo', Efo', solve_opts);
fprintf(1, 'COMPUTED IN %.2f s\n', toc(Q_start))
[XObsv, LObsv] = eig(YObsv);

% Gramian is Pfl = Zcont*Ycont*Zcont'. Too large to form as dense matrix.
% Instead:
%   1. Factor [X, L] = eig(YCont);
%   2. Based on truncation, keep leading eigenvalues
%   3. Take ZContFact = Z*X(1:t)*sqrt(L(1:t));
% And similarly for ZObsvFact
LCont          = sort(diag(LCont), 'descend');
tol            = 10e-14;
t1             = find(LCont>tol, 1,'last'); % Truncation order
LObsv          = sort(diag(LObsv), 'descend');
tol            = 10e-14;
t2             = find(LObsv>tol, 1,'last'); % Truncation order
ZContFact      = ZCont*XCont(:, 1:t1)*diag(sqrt(LCont(1:t1)));
ZObsvFact      = ZObsv*XObsv(:, 1:t2)*diag(sqrt(LObsv(1:t2)));

% Formulate projection matrices
[U, S, Y] = svd(ZObsvFact'*Efo*ZContFact);

r = min([r, t1, t2]);
% Compute projection matrices
V = ZContFact*Y(:, 1:r)*S(1:r, 1:r)^(-1/2); % Right
W = ZObsvFact*U(:, 1:r)*S(1:r, 1:r)^(-1/2); % Left


% Compute reduced order model via projection
Er_flBTExact = eye(r, r); Ar_flBTExact = W'*Afo*V;    
Br_flBTExact = W'*Bfo;    Cr_flBTExact = Cfo*V;  
Dr_flBTExact = Dfo;


fprintf(1, 'flBT REDUCED MODEL COMPUTED IN %.2f s\n', toc(flQuadBT_start))
fprintf(1, '---------------------------------------\n')

% write = 1;
% if write
% 
% else
% 
% end

%% Plot frequency response functions.
lens           = 750;                   % No. of frequencies to sample at
s              = logspace(-5, 1, lens); % Contains Omega (frequency band of interest)
res_fom        = zeros(lens, 1);        % Response of full order model
resr_flBT      = zeros(lens, 1);        % Response of (intrusive) flBT reduced model
resr_flQuadbBT = zeros(lens, 1);        % Response of (non-intrusive) flQuadBT reduced model
errr_flBT      = zeros(lens, 1);        % Error due to flBt
errr_flQuadBT  = zeros(lens, 1);        % Error due to flQuadBT

% Plot frequency response along imaginary axis
for ii=1:lens
    fprintf(1, 'Frequency step %d, s=1i*%.2f ...\n ', ii, s(ii))
    % Evaluate transfer function magnitude
    Gfo                = Cfo*((1i*s(ii)*Efo - Afo)\Bfo) + Dfo;
    Gr_flBT            = Cr_flBTExact*((1i*s(ii)*Er_flBTExact - Ar_flBTExact)\Br_flBTExact) + Dr_flBTExact;
    Gr_flQuadBT        = Cr_flQuadBT*((1i*s(ii)*Er_flQuadBT - Ar_flQuadBT)\Br_flQuadBT) + Dr_flQuadBT;
    res_fom(ii)        = norm(Gfo, 2); 
    resr_flBT(ii)      = norm(Gr_flBT, 2); 
    resr_flQuadbBT(ii) = norm(Gr_flQuadBT, 2); 
    errr_flBT(ii)      = norm(Gfo - Gr_flBT, 2); 
    errr_flQuadBT(ii)  = norm(Gfo - Gr_flQuadBT, 2); 
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

figure(2)
fs = 12;
% Magnitudes
set(gca, 'fontsize', 10)
subplot(2,1,1)
loglog(s, res_fom, '-', 'linewidth', 2, 'color', ColMat(1,:)); hold on
loglog(s, resr_flBT, '-.', 'linewidth', 2, 'color', ColMat(2,:)); 
loglog(s, resr_flQuadbBT, '--', 'linewidth', 2, 'color', ColMat(3,:)); 
leg = legend('Full-order', 'flbt', 'qflbt', 'location', 'southeast', ...
    'orientation', 'horizontal', 'interpreter', 'latex');
set(leg, 'fontsize', 10, 'interpreter', 'latex')
xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')

% Relative errors
subplot(2,1,2)
loglog(s, errr_flBT./res_fom, '-.', 'linewidth', 2, 'color', ColMat(2,:));
hold on
loglog(s, errr_flQuadBT./res_fom, '--', 'linewidth', 2, 'color', ColMat(3,:)); 
leg = legend('flBT', 'flQuadBT', 'location', 'southeast', ...
    'orientation', 'horizontal', 'interpreter', 'latex');
set(leg, 'fontsize', 10, 'interpreter', 'latex')
xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')


%% Modified Gramians.



%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off
