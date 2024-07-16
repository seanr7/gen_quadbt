%% RUNME_BUTTERFLY
% Script file to run all experiments involving the Butterfly Gyroscope
% model.

%
% This file is part of the archive Code, Data, and Results for Numerical 
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
fprintf(1, 'Loading butterfly gyroscope benchmark problem.\n')
fprintf(1, '----------------------------------------------\n')

% From: https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Butterfly_Gyroscope
% Rayleigh Damping: D = alpha*M + beta*K
%   alpha = 0;
%   beta  = 1e-6;
load('data/Butterfly.mat')


%% Frequency-limited balanced truncation from data.

% Test performance from i[1e2, 1e8]; limited frequency band of interest
% i[1e4, 1e6].
a = 4;  b = 6;  nNodes = 400;          

% Prepare quadrature weights and nodes according to Trapezoidal rule.
[nodesLeft, weightsLeft, nodesRight, weightsRight] = trapezoidal_rule([a, b], ...
    nNodes, true);

% Transfer function data.
% recomputeSamples = true;
recomputeSamples = false;
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
    WLeft   = zeros(2*n, nNodes*p);
    VRight  = zeros(2*n, nNodes*m);
    % Since we need the first-order solves for verifying that the Loewner
    % matrices are built correctly, we compute the data using the
    % first-order realization (as opposed to second-order).
    for k = 1:nNodes
        % Requisite linear solves.
        tic
        fprintf(1, 'Initiating structured linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------------------------\n');
        % Transfer function data.
        % Gives Vright = (nodesRight(k)*Efo - Afo)\Bfo.
        vRight           = so_structured_solve(M, D, K, B, C, nodesRight(k), 0);
        % Gives WLeft  = ((nodesLeft(k)*Efo - Afo)')\Cfo'.
        wLeft            = so_structured_solve(M, D, K, B, C, nodesLeft(k), 1);
        fprintf(1, 'Structured solves finished in %.2f s.\n',toc)
        fprintf(1, '-----------------------------------------------\n');

        % Transfer function data.
        GsLeft(:, :, k)              = wLeft'*Bfo;
        GsRight(:, :, k)             = Cfo * vRight;
        WLeft(:, (k - 1)*p + 1:k*p)  = wLeft;
        VRight(:, (k - 1)*m + 1:k*m) = vRight;
    end
    save('data/ButterflySamples_BandLimited.mat', 'GsLeft', 'GsRight', ...
        'nodesLeft', 'nodesRight', 'WLeft', 'VRight', '-v7.3')
else
    fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
    fprintf(1, '-------------------------------------------\n')
    load('data/ButterflySamples_BandLimited.mat', 'GsLeft', 'GsRight')
end

fprintf(1, 'BUILDING LOEWNER QUADRUPLE.\n')
fprintf(1, '---------------------------\n')
timeLoewner = tic;
% Loewner quadruple.
[Lbar, Mbar, Hbar, Gbar] = loewner_factory(nodesLeft, nodesRight, weightsLeft, ...
    weightsRight, GsLeft, GsRight);
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

% checkLoewner = true;
checkLoewner = false;
if checkLoewner
    fprintf(1, 'Sanity check: Verify that the build of the Loewner matrices is correct.\n')
    fprintf(1, '-----------------------------------------------------------------------\n')
    % Load pre-computed linear solves.
    load('data/ButterflySamples_BandLimited', 'WLeft', 'VRight')

    % Quadrature-based square root factors.
    timeFactors     = tic;
    leftObsvFactor  = zeros(2*n, nNodes*p);
    rightContFactor = zeros(2*n, nNodes*m);
    fprintf(1, 'COMPUTING APPROXIMATE SQUARE-ROOT FACTORS.\n')
    fprintf(1, '------------------------------------------\n')
    for k = 1:nNodes
        rightContFactor(:, (k - 1)*m + 1:k*m) = weightsRight(k)*VRight(:, (k - 1)*m + 1:k*m);
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*WLeft(:, (k - 1)*p + 1:k*p);
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeLoewner))
    fprintf(1, '------------------------------------------\n')

    % Build first-order system realization for check.
    % Descriptor matrix.
    Efo                       = spalloc(2*n, 2*n, nnz(M) + n); % Descriptor matrix; Efo = [I, 0: 0, M]
    Efo(1:n, 1:n)             = speye(n);                      % (1, 1) block
    Efo(n + 1:2*n, n + 1:2*n) = M;                             % (2, 2) block is (sparse) mass matrix
    % State matrix.
    Afo                       = spalloc(2*n, 2*n, nnz(K) + nnz(D) + n); % Afo = [0, I; -Kso, -Dso]
    Afo(1:n, n + 1:2*n)       = speye(n);                               % (1, 2) block of Afo
    Afo(n + 1:2*n, 1:n)       = -K;                                     % (2, 1) block is -K
    Afo(n + 1:2*n, n + 1:2*n) = -D;                                     % (2, 2) block is -D
    % Input matrix.
    Bfo               = spalloc(2*n, m, nnz(B)); % Bfo = [0; B];
    Bfo(n + 1:2*n, :) = B; 
    % Output matrix. Position input, only
    Cfo         = spalloc(p, 2*n, nnz(C)); % Cfo = [C, 0];
    Cfo(:, 1:n) = C; 

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Lbar: Error || Lbar - leftObsvFactor.H * Efo * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactor' * Efo * rightContFactor - Lbar, 2))
    fprintf('Check for Mbar: Error || Mbar - leftObsvFactor.H * Afo * rightContFactor ||_2: %.16f\n', ...
        norm(leftObsvFactor' * Afo * rightContFactor - Mbar, 2))
    fprintf('Check for Hbar: Error || Hbar - leftObsvFactor.H * Bfo                   ||_2: %.16f\n', ...
        norm(leftObsvFactor' * Bfo - Hbar, 2))
    fprintf('Check for Gbar: Error || Gbar - Cfo * rightContFactor                    ||_2: %.16f\n', ...
        norm(Cfo * rightContFactor - Gbar, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end


fprintf(1, 'COMPUTING REDUCED MODEL VIA flQuadBT (non-intrusive).\n')
fprintf(1, '-----------------------------------------------------\n')
timeflBT = tic;
% Specify order of reduction.
r = 30; 
% Reduction step. 
[Hsvs, Er_flQuadBT, Ar_flQuadBT, Br_flQuadBT, Cr_flQuadBT] = quadbt_reductor(Lbar, ...
    Mbar, Hbar, Gbar, r);
Dr_flQuadBT                                                = zeros(p, m);
fprintf(1, 'flQuadBT REDUCED MODEL COMPUTED IN %.2f s\n', toc(timeflBT))
fprintf(1, '-----------------------------------------\n')


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
    'FreqRange', [2, 8]);
opts.krylovopts.krylovWopts      = opts.krylovopts.krylovVopts;

% Frequency-limited options.
opts.mormethodopts.FreqRange        = [1.0e+04, 1.0e+06];
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

% 'Exact' intrusive using approximate factors of the fl Gramians.
fprintf(1, 'COMPUTING REDUCED MODEL VIA flBT (intrusive, no intermediate reduction).\n')
fprintf(1, '------------------------------------------------------------------------\n')
timeflBT = tic;

% For stability of solver; use strictly dissipative companion form
% Compute realization alpha.
mfun  = @(x) (M * x + 0.25 * (D * (K \ (D * x))));
alpha = 1 / (2 * eigs(mfun, n, D, 1, 'LR', struct('disp', 0)));

% Get strictly dissipative realization.
% Efo = [K, alpha * M; alpha * M, M];
Efo                   = spalloc(2*n, 2*n, nnz(K) + nnz(M));
Efo(1:n, 1:n)         = K;
Efo(1:n, n+1:2*n)     = alpha*M;      
Efo(n+1:2*n, 1:n)     = alpha*M;      
Efo(n+1:2*n, n+1:2*n) = M;      

% Afo = [-alpha * K, K - alpha * D; -K, -D + alpha * M];
Afo                   = spalloc(2*n, 2*n, nnz(K) + nnz(K - alpha*D) + nnz(K) + nnz(-D + alpha*M));
Afo(1:n, 1:n)         = -alpha*K;
Afo(1:n, n+1:2*n)     = K - alpha*D;      
Afo(n+1:2*n, 1:n)     = -K;      
Afo(n+1:2*n, n+1:2*n) = -D + alpha*M;     

% Note; Bfo and Cfo must be dense for matrix eqn solver!
% Bfo = [alpha * B; B];
Bfo             = zeros(2*n, m);
Bfo(1:n, :)     = alpha*B; 
Bfo(n+1:2*n, :) = B; 

% Cfo = [C, zeros(size(C))];
Cfo         = zeros(p, 2*n);
Cfo(:, 1:n) = C;

% Solver options.
solve_opts = struct( ...
    'FctUpdate' , 1, ...
    'Freqs'     , [1.0e+04, 1.0e+06], ...
    'Info'      , 2, ...
    'MaxIter'   , 200, ...
    'MinIter'   , 10, ...
    'ModGramian', 0, ...
    'Npts'      , 1601, ...
    'Shifts'    , 'imaginary', ...
    'Solver'    , 'logm', ...
    'SolverFreq', 1, ...
    'StoreFacE' , 0, ...
    'StoreV'    , 0, ...
    'TolComp'   , eps, ...
    'TolLyap'   , 1.0e-10, ...
    'TolRHS'    , 1.0e-12, ...
    'TrueRes'   , 1, ...
    'L'         , [], ...
    'pL'        , []);

timePstart = tic;
fprintf(1, 'SOLVING FOR FREQUENCY-LIMITED CONTROLLABILITY GRAMIAN.\n')
fprintf(1, '------------------------------------------------------\n')
[ZCont, YCont, infoCont] = freq_lyap_rksm(Afo, Bfo, Efo, solve_opts);
fprintf(1, 'COMPUTED IN %.2f s\n', toc(timePstart))
[XCont, LCont] = eig(YCont);

timeQstart = tic;
fprintf(1, 'SOLVING FOR FREQUENCY-LIMITED OBSERVABILITY GRAMIAN.\n')
fprintf(1, '----------------------------------------------------\n')
[ZObsv, YObsv, infoObsv] = freq_lyap_rksm(Afo', Cfo', Efo', solve_opts);
fprintf(1, 'COMPUTED IN %.2f s\n', toc(timeQstart))
[XObsv, LObsv] = eig(YObsv);

% Gramian is Pfl = Zcont*Ycont*Zcont'. Too large to form as dense matrix.
% Instead:
%  1. Factor [X, L] = eig(YCont);
%  2. Based on truncation, keep leading eigenvalues
%  3. Take ZContFact = Z*X(1:t)*sqrt(L(1:t));
% And similarly for ZObsvFact
[LCont, p1]    = sort(diag(LCont), 'descend');
tol            = 10e-14;
t1             = find(LCont>tol, 1,'last'); % Truncation order
[LObsv, p2]    = sort(diag(LObsv), 'descend');
tol            = 10e-14;
t2             = find(LObsv>tol, 1,'last'); % Truncation order
ZContFact      = ZCont*XCont(:, p1(1:t1))*diag(sqrt(LCont(1:t1)));
ZObsvFact      = ZObsv*XObsv(:, p2(1:t2))*diag(sqrt(LObsv(1:t2)));

% Formulate projection matrices
[U, S, Y] = svd(ZObsvFact'*Efo*ZContFact);

r = min([r, t1, t2]);
% Compute projection matrices
V = ZContFact*Y(:, 1:r)*S(1:r, 1:r)^(-1/2); % Right
W = ZObsvFact*U(:, 1:r)*S(1:r, 1:r)^(-1/2); % Left

% Compute reduced order model via projection
Er_flBTExact = eye(r, r); Ar_flBTExact = W'*Afo*V;    
Br_flBTExact = W'*Bfo;    Cr_flBTExact = Cfo*V;  
Dr_flBTExact = zeros(p, m);

fprintf(1, 'flBT REDUCED MODEL COMPUTED IN %.2f s\n', toc(timeflBT))
fprintf(1, '------------------------------------------------\n')

%% Plots.
numSamples     = 500;
s              = 1i*logspace(2, 8, numSamples);
flQuadBTResp   = zeros(numSamples, 1);          % Response of (non-intrusive) flQuadBT reduced model
flQuadBTError  = zeros(numSamples, 1);          % Error due to (non-intrusive) flQuadBT reduced model
flBTInterResp  = zeros(numSamples, 1);          % Response of (intrusive, intermediate reduction) flBT reduced model
flBTInterError = zeros(numSamples, 1);          % Error due to (intrusive, intermediate reduction) flBT reduced model
flBTExactResp  = zeros(numSamples, 1);          % Response of (intrusive, intermediate reduction) flBT reduced model
flBTExactError = zeros(numSamples, 1);          % Error due to (intrusive, intermediate reduction) flBT reduced model

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
    save('data/ButterflyFullOrderSimData.mat', 'Gfo', 'GfoResp')
else
    fprintf(1, 'Loading precomputed values.\n')
    fprintf(1, '--------------------------------------------------------\n')
    load('data/ButterflyFullOrderSimData.mat')
end


% Plot frequency response along imaginary axis.
for ii=1:numSamples
    fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, s(ii))
    % Evaluate transfer function magnitude
    Gr_flQuadBT        = Cr_flQuadBT*((s(ii)*Er_flQuadBT - Ar_flQuadBT)\Br_flQuadBT) ...
        + Dr_flQuadBT;
    Gr_flBTInter       = Cr_flBTInter*((s(ii)*Er_flBTInter - Ar_flBTInter)\Br_flBTInter) ...
        + Dr_flBTInter;
    Gr_flBTExact       = Cr_flBTExact*((s(ii)*Er_flBTExact - Ar_flBTExact)\Br_flBTExact) ...
        + Dr_flBTExact;
    flQuadBTResp(ii)   = norm(Gr_flQuadBT, 2); 
    flQuadBTError(ii)  = norm(Gfo(:, :, ii) - Gr_flQuadBT, 2); 
    flBTInterResp(ii)  = norm(Gr_flBTInter, 2); 
    flBTInterError(ii) = norm(Gfo(:, :, ii) - Gr_flBTInter, 2); 
    flBTExactResp(ii)  = norm(Gr_flBTExact, 2); 
    flBTExactError(ii) = norm(Gfo(:, :, ii) - Gr_flBTExact, 2); 
    fprintf(1, '----------------------------------------------------------------------\n');
end

% Plot colors
ColMat = zeros(6,3);
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
loglog(imag(s), flBTExactResp, '-.', 'linewidth', 2, 'color', ColMat(4,:)); 
leg = legend('Full-order', 'flQuadBT', 'flBTInter', 'flBTExact', 'location', 'southeast', 'orientation', 'horizontal', ...
    'interpreter', 'latex');
xlim([imag(s(1)), imag(s(end))])
set(leg, 'fontsize', 10, 'interpreter', 'latex')
xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')

% Relative errors
subplot(2,1,2)
loglog(imag(s), flQuadBTError./GfoResp,  '-o', 'linewidth', 2, 'color', ColMat(2,:)); hold on
loglog(imag(s), flBTInterError./GfoResp, '-*', 'linewidth', 2, 'color', ColMat(3,:));
loglog(imag(s), flBTExactError./GfoResp, '-*', 'linewidth', 2, 'color', ColMat(4,:));
leg = legend('flQuadBT', 'flBTInter', 'flBTExact', 'location', 'southeast', 'orientation', 'horizontal', ...
    'interpreter', 'latex');
xlim([imag(s(1)), imag(s(end))])
set(leg, 'fontsize', 10, 'interpreter', 'latex')
xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', ...
    fs, 'interpreter', 'latex')


%% Modified Gramians.
 


%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off