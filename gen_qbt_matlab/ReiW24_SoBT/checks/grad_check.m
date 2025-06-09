%% GRAD_TEST
% Script to test minimization routine for damping parameters.
%

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


%% Toy problem.
fprintf(1, 'Loading MSD benchmark problem.\n')
fprintf(1, '----------------------------------------------\n')

n1    = 100;   
alpha = .002;   
beta  = alpha;  
v     = 0;
[M, D, K] = triplechain_MSD(n1, alpha, beta, v);

% Rayleigh damping.
M = full(M); D = full(D); K = full(K);

% Input, output, state dimensions.
n = size(full(K), 1);   p = 1;  m = 1;

% Input and output matrices.
B   = ones(n, m);   
Cp  = ones(p, n);

% 1. Generate `true' transfer function data (using true alpha, beta) to 
%    do least squares fit against.
% 2. Compute reduced-order model via soQuadBt with true data.
% 3. With `true' data and reduced-order Mr, Kr, Br, Cpr, optimize over
%    (unknown) alpha, beta, to minimize the least squares fit. 
%   3a. Check gradients numerically against finite differences!
%   3b. Minimize.

%% Test gradients for Rayleigh damping. 

%% 1. Generate true data. 
% Test performance from i[1e-3, 1e0]; interesting response behavior for MSD
% problem is here.
a = -3;  b = 0;  nNodes = 100;      

% Prepare quadrature weights and nodes according to Trapezoidal rule.
[nodesLeft, weightsLeft, nodesRight, weightsRight] = trapezoidal_rule([a, b], ...
    nNodes, true);

% Put into complex conjugate pairs to make reduced-order model matrices
% real valued. 
[nodesLeft, Ileft]   = sort(nodesLeft, 'ascend');    
[nodesRight, Iright] = sort(nodesRight, 'ascend');   
weightsLeft          = weightsLeft(Ileft);
weightsRight         = weightsRight(Iright);

% Generate true data.
GsLeft  = zeros(p, m, nNodes);
GsRight = zeros(p, m, nNodes);
for k = 1:nNodes
    % Requisite linear solves.
    tic
    fprintf(1, 'Linear solve %d of %d.\n', k, nNodes)
    fprintf(1, '-----------------------------\n');
    % Transfer function data.
    GsRight(:, :, k) = Cp*((nodesRight(k)^2.*M + nodesRight(k).*D + K)\B);
    GsLeft(:, :, k)  = Cp*((nodesLeft(k)^2.*M  + nodesLeft(k).*D  + K)\B);
    fprintf(1, 'Solves finished in %.2f s.\n',toc)
    fprintf(1, '-----------------------------\n');
end
    
%% 2. Compute reduced-order model via soQuadBt with true data.
r = 10;
fprintf(1, 'BUILDING LOEWNER QUADRUPLE (soQuadBT).\n')
fprintf(1, '--------------------------------------\n')
timeLoewner = tic;

% Loewner quadruple.
[Mbar_soQuadBT, ~, Kbar_soQuadBT, Bbar_soQuadBT, CpBar_soQuadBT] = ...
    so_loewner_factory(nodesLeft, nodesRight, weightsLeft, weightsRight, GsLeft, ...
                       GsRight, 'Rayleigh', [alpha, beta], 'Position');
fprintf(1, 'CONSTRUCTION OF LOEWNER QUADRUPLE FINISHED IN %.2f s\n', toc(timeLoewner))
fprintf(1, '------------------------------------------------------\n')

% Make it real-valued.
J = zeros(nNodes, nNodes);
for i = 1:nNodes/2
    J(1 + 2*(i - 1):2*i, 1 + 2*(i - 1):2*i) = 1/sqrt(2)*[1, -1i; 1, 1i];
end

Mbar_soQuadBT = J'*Mbar_soQuadBT*J;   Kbar_soQuadBT  = J'*Kbar_soQuadBT*J;   
Mbar_soQuadBT = real(Mbar_soQuadBT);  Kbar_soQuadBT  = real(Kbar_soQuadBT);  
Bbar_soQuadBT = J'*Bbar_soQuadBT;      CpBar_soQuadBT = CpBar_soQuadBT*J;
Bbar_soQuadBT = real(Bbar_soQuadBT);  CpBar_soQuadBT = real(CpBar_soQuadBT);

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
        % For (position) controllability Gramian.
        rightContFactor(:, (k - 1)*m + 1:k*m) = weightsRight(k)*(((nodesRight(k)^2.*M ...
            + nodesRight(k).*D + K)\B));

        % For (velocity) observability Gramian.
        leftObsvFactor(:, (k - 1)*p + 1:k*p)  = conj(weightsLeft(k))*(((conj(nodesLeft(k))^2.*M' ...
            + conj(nodesLeft(k)).*D' + K')\Cp'));
    end
    fprintf(1, 'APPROXIMATE FACTORS COMPUTED IN %.2f s\n', toc(timeFactors))
    fprintf(1, '------------------------------------------\n')

    fprintf('-----------------------------------------------------------------------------------\n')
    fprintf('Check for Mbar  : Error || Mbar  - leftObsvFactor.H * M * rightContFactor ||_2: %.16f\n', ...
        norm(J'*leftObsvFactor'*M*rightContFactor*J - Mbar_soQuadBT, 2))
    fprintf('Check for Dbar  : Error || Dbar  - leftObsvFactor.H * D * rightContFactor ||_2: %.16f\n', ...
        norm(J'*leftObsvFactor'*D*rightContFactor*J - Dbar_soQuadBT, 2))
    fprintf('Check for Kbar  : Error || Kbar  - leftObsvFactor.H * K * rightContFactor ||_2: %.16f\n', ...
        norm(J'*leftObsvFactor'*K*rightContFactor*J - Kbar_soQuadBT, 2))
    fprintf('Check for Bbar  : Error || Bbar  - leftObsvFactor.H * B                   ||_2: %.16f\n', ...
        norm(J'*leftObsvFactor'*B - Bbar_soQuadBT, 2))
    fprintf('Check for CpBar : Error || CpBar - Cp * rightContFactor                   ||_2: %.16f\n', ...
        norm(Cp*rightContFactor*J - CpBar_soQuadBT, 2))
    fprintf('-----------------------------------------------------------------------------------\n')
else
    fprintf(1, 'Not verifying Loewner build; moving on.\n')
    fprintf(1, '---------------------------------------\n')
end

% Reductor step.
[Z_soQuadBT, S_soQuadBT, Y_soQuadBT] = svd(Mbar_soQuadBT);

% Enforce mass to be identity. 
Mr_soQuadBT  = eye(r, r);
Kr_soQuadBT  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Kbar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
Dr_soQuadBT  = alpha*Mr_soQuadBT + beta*Kr_soQuadBT;
Cpr_soQuadBT = CpBar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
Br_soQuadBT  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Bbar_soQuadBT;


%% 3. 
%  With `true' data and reduced-order Mr, Kr, Br, Cpr, optimize over
%  (unknown) alpha, beta, to minimize the least squares fit. 

% Options for fminunc.
options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton', ...
    'SpecifyObjectiveGradient', true);
    % 'checkGradients', true);

% Initial values for optimization variables (how close to put to true)
alpha0 = 1;    beta0  = 1;
fprintf(1, 'INITIAL VALUES FOR OPTIMIZATION VARIABLES.\n')
fprintf(1, 'alpha0 = %d\n', alpha0)
fprintf(1, 'beta0  = %d.\n', beta0)
initDampingParams = [alpha0, beta0];

% Note: This is how to create a symbolic function (one symbolic input,
%       vector-valued in this case) with other fixed inputs.
% Create instance of objective function to pass to the solver.
objFunc = @(dampingParams) rayleigh_damping_obj(dampingParams, [nodesLeft; nodesRight], ...
    [cat(3, GsLeft, GsRight)], Kr_soQuadBT, Br_soQuadBT, Cpr_soQuadBT, zeros(p, r));


%% 3a.
% Test gradients against finite differences.
% Evaluate gradient (at initial points).
[val, grads] = objFunc(initDampingParams);
gradAlpha = grads(1);
gradBeta  = grads(2);
pertAlpha = rand;
pertBeta  = rand; 

% Evaluate objective function for perturbed parameters.
fprintf(1, 'GRADIENT WITH RESPECT TO B\n')
fprintf(1, '--------------------------\n')
for eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

    % Grad w.r.t. alpha
    % Evaluate J(x + dx), J(x - dx)
    pertErrorPlus_Alpha  = objFunc([alpha0 + pertAlpha*eps, beta0]);
    pertErrorMinus_Alpha = objFunc([alpha0 - pertAlpha*eps, beta0]);
    fprintf(1, 'FINITE DIFFERENCE ESTIMATE: %.16f\n', (pertErrorPlus_Alpha - pertErrorMinus_Alpha)/(2*eps))
    fprintf(1, 'ANALYTIC GRADIENT (alpha) : %.16f\n', trace(gradAlpha'*(pertAlpha)))
    fprintf(1, 'REL. ERROR (alpha)        : %.16f\n', abs(gradAlpha'*(pertAlpha) - ((pertErrorPlus_Alpha - ...
        pertErrorMinus_Alpha))/(2*eps))/abs(gradAlpha'*(pertAlpha)))
    fprintf(1, '-------------------------------------------------\n')

    % Grad w.r.t. beta
    % Evaluate J(x + dx), J(x - dx)
    pertErrorPlus_Beta  = objFunc([alpha0, beta0 + pertBeta*eps]);
    pertErrorMinus_Beta = objFunc([alpha0, beta0 - pertBeta*eps]);
    fprintf(1, 'FINITE DIFFERENCE ESTIMATE: %.16f\n', (pertErrorPlus_Beta - pertErrorMinus_Beta)/(2*eps))
    fprintf(1, 'ANALYTIC GRADIENT (beta)  : %.16f\n', trace(gradBeta'*(pertBeta)))
    fprintf(1, 'REL. ERROR (beta)         : %.16f\n', abs(gradBeta'*(pertBeta) - ((pertErrorPlus_Beta - ...
        pertErrorMinus_Beta))/(2*eps))/abs(gradBeta'*(pertBeta)))
    fprintf(1, '-------------------------------------------------\n')
end

% Also, lets plot the objective function.
numTestPoints = 500;

% Because `true' parameters are alpha = beta = 2e-3, plot around there.
testPoints = 2*logspace(-4, 0, numTestPoints);
objAlpha   = zeros(numTestPoints, 1);
objBeta    = zeros(numTestPoints, 1);

% Evaluate objective function for different values of alpha, beta. about
% alpha0, beta0.
for ii = 1:numTestPoints
    objAlpha(ii) = objFunc([testPoints(ii), beta0]);
    objBeta(ii)  = objFunc([alpha0,         testPoints(ii)]);
end

figure
subplot(2, 1, 1)
loglog(testPoints, objAlpha, '--', 'linewidth', 2)
title('Changes in objFunc w.r.t. alpha')
xlabel('alpha')
xlim([testPoints(1), testPoints(end)])

subplot(2, 1, 2)
loglog(testPoints, objBeta, '--', 'linewidth', 2)
title('Changes in objFunc w.r.t. beta')
xlabel('beta')
xlim([testPoints(1), testPoints(end)])


%% 3b.
% Minimizer.
optDampingParams = fminunc(objFunc, initDampingParams, options);
optAlpha         = optDampingParams(1);
optBeta          = optDampingParams(1);

fprintf(1, 'DIFFERENCE IN FOUND DAMPING PARAMETERS COMPARED TO TRUE.\n')
fprintf(1, '|alpha - optAlpha|: %.16f\n', abs(alpha-optAlpha))
fprintf(1, '|beta  - optbeta |: %.16f\n', abs(beta-optBeta))
fprintf(1, '--------------------------------------------------------\n')

% Now, damping with `optimal' parameters.
Dr_soQuadBT_optParams = optAlpha*Mr_soQuadBT + optBeta*Kr_soQuadBT;

%% Resulting model fit.
% Plots.
numSamples     = 500;
s              = 1i*logspace(a, b, numSamples);
soQuadBTResp   = zeros(numSamples, 1);          % Response of (non-intrusive) soQuadBT reduced model with true damping params
soQuadBTError  = zeros(numSamples, 1);          % Error due to (non-intrusive) soQuadBT reduced model with true damping params

soQuadBTResp_optParams   = zeros(numSamples, 1); % Response of (non-intrusive) soQuadBT reduced model with computed damping params
soQuadBTError_optParams  = zeros(numSamples, 1); % Error due to (non-intrusive) soQuadBT reduced model with computed damping params

% Full-order simulation data.
Gfo     = zeros(p, m, numSamples);
GfoResp = zeros(numSamples, 1);
GfoFrob = zeros(numSamples, 1);
fprintf(1, 'Sampling full-order transfer function along i[1e-3, 1e0].\n')
fprintf(1, '--------------------------------------------------------\n')
for ii = 1:numSamples
    timeSolve = tic;
    % fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, s(ii))
    Gfo(:, :, ii) = Cp*((s(ii)^2*M +s(ii)*D + K)\B);
    GfoResp(ii)   = max(svd(Gfo(:, :, ii)));        % Matrix 2-norm
end

% Reduced-order data.
for ii=1:numSamples
    % fprintf(1, 'Frequency step %d, s=1i*%.10f ...\n ', ii, s(ii))
    % Evaluate transfer function magnitude
    Gr_soQuadBT           = Cpr_soQuadBT*((s(ii)^2*Mr_soQuadBT + s(ii)*Dr_soQuadBT + Kr_soQuadBT)\Br_soQuadBT);
    Gr_soQuadBT_optParams = Cpr_soQuadBT*((s(ii)^2*Mr_soQuadBT + s(ii)*Dr_soQuadBT_optParams + Kr_soQuadBT)\Br_soQuadBT);
    % soQuadBTResp(ii)   = norm(Gr_soQuadBT, 2); 
    soQuadBTResp(ii)   = max(svd(Gr_soQuadBT));
    soQuadBTError(ii)  = max(svd(Gfo(:, :, ii) - Gr_soQuadBT))/GfoResp(ii);

    soQuadBTResp_optParams(ii)   = max(svd(Gr_soQuadBT_optParams));
    soQuadBTError_optParams(ii)  = max(svd(Gfo(:, :, ii) - Gr_soQuadBT_optParams))/GfoResp(ii);
    % soQuadBTError(ii)  = norm(Gfo(:, :, ii) - Gr_soQuadBT, 2)/GfoResp(ii); 
    % fprintf(1, '----------------------------------------------------------------------\n');
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
    
    figure
    fs = 12;
    % Magnitudes
    set(gca, 'fontsize', 10)
    subplot(2,1,1)
    loglog(imag(s), GfoResp,       '-o', 'linewidth', 2, 'color', ColMat(1,:)); hold on
    loglog(imag(s), soQuadBTResp,  '--', 'linewidth', 2, 'color', ColMat(2,:)); 
    loglog(imag(s), soQuadBTResp_optParams,  '--', 'linewidth', 2, 'color', ColMat(2,:)); 
    leg = legend('Full-order', 'soQuadBT (true damping)', 'soQuadBT (optimal/computed damping)', 'location', 'southeast', 'orientation', 'horizontal', ...
        'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)||_2$', 'fontsize', fs, 'interpreter', 'latex')
    
    % Relative errors
    subplot(2,1,2)
    loglog(imag(s), soQuadBTError./GfoResp,  '--', 'linewidth', 2, 'color', ColMat(2,:)); hold on
    loglog(imag(s), soQuadBTError_optParams./GfoResp,  '-.', 'linewidth', 2, 'color', ColMat(3,:)); 
    leg = legend('soQuadBT (true damping)', 'soQuadBT (optimal/computed damping)', 'location', 'southeast', 'orientation', 'horizontal', ...
        'interpreter', 'latex');
    xlim([imag(s(1)), imag(s(end))])
    set(leg, 'fontsize', 10, 'interpreter', 'latex')
    xlabel('$i*\omega$', 'fontsize', fs, 'interpreter', 'latex')
    ylabel('$||\mathbf{G}(s)-\mathbf{G}_{r}(s)||_2/||\mathbf{G}(s)||_2$', 'fontsize', ...
        fs, 'interpreter', 'latex')
end

%% Finished script.
fprintf(1, 'FINISHED SCRIPT.\n');
fprintf(1, '================\n');
fprintf(1, '\n');

diary off

