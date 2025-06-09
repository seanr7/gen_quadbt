%% LOEWNER_CHECK
% Script to test Loewner builds.

clc;
clear all;
close all;

% Get and set all paths
[rootpath, filename, ~] = fileparts(mfilename('fullpath'));
loadname            = [rootpath filesep() ...
    'data' filesep() filename];
savename            = [rootpath(1:end-7) filesep() ...
    'checks_results' filesep() filename];

% Add paths to drivers and data
addpath([rootpath(1:end-7), '/drivers'])
addpath([rootpath(1:end-7), '/data'])
addpath([rootpath(1:end-7), '/results'])

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


%% 1. Rayleigh damping.
fprintf(1, '1. RAYLEIGH DAMPED PROBLEM.\n')
fprintf(1, '----------------------------------------------\n')


%% Problem data.
fprintf(1, 'Loading butterfly gyroscope benchmark problem.\n')
fprintf(1, '----------------------------------------------\n')

% From: https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Butterfly_Gyroscope
% Rayleigh Damping: D = alpha*M + beta*K
%   alpha = 0;
%   beta  = 1e-6;
load('data/Butterfly.mat')

% Rayleigh damping coefficients.
alpha = 0;
beta  = 1e-6;

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

% Transfer function data.
fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
fprintf(1, '-------------------------------------------\n')
load('butterfly_samples_N200_1e4to1e6.mat', 'GsLeft', 'GsRight')

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
        solveTime = tic;
        fprintf(1, 'Linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k)^2.*M ...
            + nodesRight(k).*D + K)\B));
    
        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k))^2.*M' ...
            + conj(nodesLeft(k)).*D' + K')\C'));

        fprintf(1, 'Solves finished in %.2f s.\n', toc(solveTime))
        fprintf(1, '-----------------------------\n');
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')
    
    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Mbar  : Error || Mbar  - leftObsvFactor.H * M * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*M*rightContFactor*Jm - Mbar_soQuadBT, 2))
    fprintf('Check for Dbar  : Error || Dbar  - leftObsvFactor.H * D * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*D*rightContFactor*Jm - Dbar_soQuadBT, 2))
    fprintf('Check for Kbar  : Error || Kbar  - leftObsvFactor.H * K * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*K*rightContFactor*Jm - Kbar_soQuadBT, 2))
    fprintf('Check for Bbar  : Error || Bbar  - leftObsvFactor.H * B                   ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*B - Bbar_soQuadBT, 2))
    fprintf('Check for CpBar : Error || CpBar - Cp * rightContFactor                   ||_2: %.16f\n', ...
        norm(C*rightContFactor*Jm - CpBar_soQuadBT, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

fprintf(1, 'BUILDING LOEWNER MATRICES (soLoewner).\n')
fprintf(1, '---------------------------------------\n')
timeLoewner = tic;
% Note: This construction is a diagonally scaled (equivalent) realization
% of then one in [Def. 2, Thm. 3, PonGB22]. See Remark 3.3 in the companion
% paper for details.
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

% See note below regarding this check, and why
% it fails for the precise construction from [Def. 2, PonGB22]. 
    % Note; using our construction (from the rational Krylov
    % subspaces) we get a different formulation than the one from
    % [Def. 2, PonGB22]. Specifically, the denominator scaling
    % should still be there. 
    % So, this is making the Loewner check fail; if you use the
    % first option, its good. If you use the second, it fails. But,
    % the second is what is in [PonGB22], so we use that for
    % testing. Not sure if this is a mistake on their part, or some
    % change of variable.
    % Ah yes! Actually, they do have the denominator scaling. Its
    % just in B and C, so we go with the latter.

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
        solveTime = tic;
        fprintf(1, 'Linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = (((nodesRight(k)^2.*M ...
            + nodesRight(k).*D + K)\B));
    
        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = (((conj(nodesLeft(k))^2.*M' ...
            + conj(nodesLeft(k)).*D' + K')\C'));

        fprintf(1, 'Solves finished in %.2f s.\n', toc(solveTime))
        fprintf(1, '-----------------------------\n');

    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')
    
    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Mbar  : Error || Mbar  - leftObsvFactor.H * M * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*M*rightContFactor*Jm - Mbar_soLoewner, 2))
    fprintf('Check for Dbar  : Error || Dbar  - leftObsvFactor.H * D * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*D*rightContFactor*Jm - Dbar_soLoewner, 2))
    fprintf('Check for Kbar  : Error || Kbar  - leftObsvFactor.H * K * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*K*rightContFactor*Jm - Kbar_soLoewner, 2))
    fprintf('Check for Bbar  : Error || Bbar  - leftObsvFactor.H * B                   ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*B - Bbar_soLoewner, 2))
    fprintf('Check for CpBar : Error || CpBar - Cp * rightContFactor                   ||_2: %.16f\n', ...
        norm(C*rightContFactor*Jm - CpBar_soLoewner, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

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
    
    Cfo         = spalloc(p, 2*n, nnz(C)); % Cfo = [Cp, 0];
    Cfo(:, 1:n) = C; 
    
    
    for k = 1:nNodes
        solveTime = tic;
        fprintf(1, 'Linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k).*Efo - Afo)\Bfo));
    
        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k)).*Efo' - Afo')\Cfo'));
        fprintf(1, 'Solves finished in %.2f s.\n', toc(solveTime))
        fprintf(1, '-----------------------------\n');
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')
    
    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Ebar : Error || Ebar  - leftObsvFactor.H * Efo * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*Efo*rightContFactor*Jm - Ebar_foQuadBT, 2))
    fprintf('Check for Abar : Error || Abar  - leftObsvFactor.H * Afo * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*Afo*rightContFactor*Jm - Abar_foQuadBT, 2))
    fprintf('Check for Bbar : Error || Bbar  - leftObsvFactor.H * Bfo                   ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*Bfo - Bbar_foQuadBT, 2))
    fprintf('Check for Cbar : Error || Cbar - C * rightContFactor                       ||_2: %.16f\n', ...
        norm(Cfo*rightContFactor*Jm - Cbar_foQuadBT, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

%% 2. Structural damping.
fprintf(1, '1. STRUCTURALLY DAMPED PROBLEM.\n')
fprintf(1, '----------------------------------------------\n')

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
% Test performance from 1 to 251 Hz.
% Frequencies used in the simulation.
%   s    = 1i*linspace(1, 2*pi*251, 250); 
%   s_hz = imag(s)/2/pi; 
nNodes = 250;          

% Prepare quadrature weights and nodes according to Trapezoidal rule.
omega      = 1i*(linspace(1, 2*pi*251, nNodes)');
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

% Transfer function data.
fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
fprintf(1, '-------------------------------------------\n')
load('plateTVA_samples_N250_0to250Hz.mat', 'GsLeft', 'GsRight')

fprintf(1, 'BUILDING LOEWNER QUADRUPLE (soQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;

% Loewner quadruple.
[Mbar_soQuadBT, Dbar_soQuadBT, Kbar_soQuadBT, Bbar_soQuadBT, CpBar_soQuadBT] = ...
    so_loewner_factory(nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeft, ...
                       GsRight, 'Structural', eta, 'Position');
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

checkLoewner = true;
% checkLoewner = false;
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
        solveTime = tic;
        fprintf(1, 'Linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k)^2.*M ...
            + D + K)\B));
    
        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k))^2.*M' ...
            + D' + K')\Cp'));
        fprintf(1, 'Solves finished in %.2f s.\n', toc(solveTime))
        fprintf(1, '-----------------------------\n');
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
        norm(Cp * rightContFactor - CpBar_soQuadBT, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

fprintf(1, 'BUILDING LOEWNER QUADRUPLE (soLoewner).\n')
fprintf(1, '---------------------------------------\n')
timeLoewner = tic;
[Mbar_soLoewner, Dbar_soLoewner, Kbar_soLoewner, Bbar_soLoewner, CpBar_soLoewner] = ...
    so_loewner_factory(nodesLeft, nodesRight, ones(nNodes, 1), ones(nNodes, 1), GsLeft, ...
                       GsRight, 'Structural', eta, 'Position');
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

checkLoewner = true;
% checkLoewner = false;
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
        solveTime = tic;
        fprintf(1, 'Linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = (((nodesRight(k)^2.*M ...
            + D + K)\B));
    
        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = (((conj(nodesLeft(k))^2.*M' ...
            + D' + K')\Cp'));
        fprintf(1, 'Solves finished in %.2f s.\n', toc(solveTime))
        fprintf(1, '-----------------------------\n');
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
        norm(Cp * rightContFactor - CpBar_soLoewner, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

fprintf(1, 'BUILDING LOEWNER QUADRUPLE (foQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;
% Loewner quadruple.
[Ebar_foQuadBT, Abar_foQuadBT, Bbar_foQuadBT, Cbar_foQuadBT] = fo_loewner_factory(...
    nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeft, GsRight);
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

checkLoewner = true;
% checkLoewner = false;
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
    
    Bfo             = spalloc(2*n, m, nnz(B)); % Bfo = [0; B];
    Bfo(n+1:2*n, :) = B; 
    
    Cfo         = spalloc(p, 2*n, nnz(Cp)); % Bfo = [Cp, 0];
    Cfo(:, 1:n) = Cp; 
    
    
    for k = 1:nNodes
        solveTime = tic;
        fprintf(1, 'Linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k).*Efo - Afo)\Bfo));
    
        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k)).*Efo' - Afo')\Cfo'));
        fprintf(1, 'Solves finished in %.2f s.\n', toc(solveTime))
        fprintf(1, '-----------------------------\n');
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

%% 3. Velocity outputs.
fprintf(1, '3a. VELOCITY OUTPUTS (Rayleigh damped).\n')
fprintf(1, '----------------------------------------------\n')

%% Problem data.
fprintf(1, 'Loading mass-spring-damper problem.\n')
fprintf(1, '-----------------------------------\n');

% From: [Truhar and Veselic 2009, Ex. 2]
% Rayleigh Damping: D = alpha*M + beta*K
%   alpha = 2e-3;
%   beta  = 2e-3;
load('data/MSDRayleigh_Cv.mat')
alpha = 2e-3;
beta  = alpha;
D     = alpha*M + beta*K;

fprintf(1, 'LOADING PRECOMPUTED TRANSFER FUNCTION DATA.\n')
fprintf(1, '-------------------------------------------\n')
load('results/MSD_Cv_samples_N200_1e-3to1e1.mat', 'GsLeft', 'GsRight')

%% Reduced order models.
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

checkLoewner = true;
% checkLoewner = false;
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
        solveTime = tic;
        fprintf(1, 'Linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k)^2.*M ...
            + nodesRight(k).*D + K)\B));
    
        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k))^2.*M' ...
            + conj(nodesLeft(k)).*D' + K')\(conj(nodesLeft(k)).*(Cv)')));
        fprintf(1, 'Solves finished in %.2f s.\n', toc(solveTime))
        fprintf(1, '-----------------------------\n');
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')
    
    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Mbar  : Error || Mbar  - leftObsvFactor.H * M * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*M*rightContFactor*Jm - Mbar_soQuadBT, 2)/norm(Jp'*leftObsvFactor'*M*rightContFactor*Jm, 2))
    fprintf('Check for Dbar  : Error || Dbar  - leftObsvFactor.H * D * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*D*rightContFactor*Jm - Dbar_soQuadBT, 2)/norm(Jp'*leftObsvFactor'*D*rightContFactor*Jm, 2))
    fprintf('Check for Kbar  : Error || Kbar  - leftObsvFactor.H * K * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*K*rightContFactor*Jm - Kbar_soQuadBT, 2)/norm(Jp'*leftObsvFactor'*K*rightContFactor*Jm, 2))
    fprintf('Check for Bbar  : Error || Bbar  - leftObsvFactor.H * B                   ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*B - Bbar_soQuadBT, 2)/norm(Jp'*leftObsvFactor'*B, 2))
    fprintf('Check for CvBar : Error || CvBar - Cv * rightContFactor                   ||_2: %.16f\n', ...
        norm(Cv*rightContFactor*Jm - CvBar_soQuadBT, 2)/norm(Cv*rightContFactor*Jm, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

fprintf(1, 'BUILDING LOEWNER MATRICES (soLoewner).\n')
fprintf(1, '---------------------------------------\n')
timeLoewner = tic;
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

checkLoewner = true;
% checkLoewner = false;
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
        solveTime = tic;
        fprintf(1, 'Linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = ((nodesRight(k)^2.*M ...
            + nodesRight(k).*D + K)\B);
    
        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = ((conj(nodesLeft(k))^2.*M' ...
            + conj(nodesLeft(k)).*D' + K')\(conj(nodesLeft(k))*(Cv)'));
        fprintf(1, 'Solves finished in %.2f s.\n', toc(solveTime))
        fprintf(1, '-----------------------------\n');
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')
    
    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Mbar  : Error || Mbar  - leftObsvFactor.H * M * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*M*rightContFactor*Jm - Mbar_soLoewner, 2)/norm(Jp'*leftObsvFactor'*M*rightContFactor*Jm, 2))
    fprintf('Check for Dbar  : Error || Dbar  - leftObsvFactor.H * D * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*D*rightContFactor*Jm - Dbar_soLoewner, 2)/norm(Jp'*leftObsvFactor'*D*rightContFactor*Jm, 2))
    fprintf('Check for Kbar  : Error || Kbar  - leftObsvFactor.H * K * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*K*rightContFactor*Jm - Kbar_soLoewner, 2)/norm(Jp'*leftObsvFactor'*K*rightContFactor*Jm, 2))
    fprintf('Check for Bbar  : Error || Bbar  - leftObsvFactor.H * B                   ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*B - Bbar_soLoewner, 2)/norm(Jp'*leftObsvFactor'*B, 2))
    fprintf('Check for CvBar : Error || CvBar - Cv * rightContFactor                   ||_2: %.16f\n', ...
        norm(Cv*rightContFactor*Jm - CvBar_soLoewner, 2)/norm(Cv*rightContFactor*Jm, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

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

checkLoewner = true;
% checkLoewner = false;
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

    Cfo             = spalloc(1, 2*n, nnz(Cv)); % Cfo = [0, Cv];
    Cfo(:, n+1:2*n) = Cv; 


    for k = 1:nNodes
        solveTime = tic;
        fprintf(1, 'Linear solves %d of %d.\n', k, nNodes)
        fprintf(1, '-----------------------------\n');
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k).*Efo - Afo)\Bfo));

        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k)).*Efo' - Afo')\(Cfo')));
        fprintf(1, 'Solves finished in %.2f s.\n', toc(solveTime))
        fprintf(1, '-----------------------------\n');
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Ebar : Error || Ebar  - leftObsvFactor.H * Efo * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*Efo*rightContFactor*Jm - Ebar_foQuadBT, 2)/norm(Jp'*leftObsvFactor'*Efo*rightContFactor*Jm, 2))
    fprintf('Check for Abar : Error || Abar  - leftObsvFactor.H * Afo * rightContFactor ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*Afo*rightContFactor*Jm - Abar_foQuadBT, 2)/norm(Jp'*leftObsvFactor'*Afo*rightContFactor*Jm, 2))
    fprintf('Check for Bbar : Error || Bbar  - leftObsvFactor.H * Bfo                   ||_2: %.16f\n', ...
        norm(Jp'*leftObsvFactor'*Bfo - Bbar_foQuadBT, 2)/norm(Jp'*leftObsvFactor'*Bfo, 2))
    fprintf('Check for Cbar : Error || Cbar  - C * rightContFactor                       ||_2: %.16f\n', ...
        norm(Cfo*rightContFactor*Jm - Cbar_foQuadBT, 2)/norm(Cfo*rightContFactor*Jm, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

%%
fprintf(1, '3b. VELOCITY OUTPUTS (Structurally damped).\n')
fprintf(1, '----------------------------------------------\n')


%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off