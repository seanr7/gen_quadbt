% Make sure you have the necessary directory added to your MATLAB path
addpath('/Users/seanr/Desktop/gen_quadbt/gen_qbt_matlab')
addpath('/Users/seanr/Desktop/gen_quadbt/gen_qbt_matlab/benchmarks/')
clear all

%% 1. Load benchmark for testing 
% Set mass, spring, damping coefficients
n = 400;
[A, B, C, D, ~]=circuit(n);

sampler = QuadPRBTSampler(A, B, C, D);

%% 2. Instantiate reductor + build Loewner quadruple (Lbar, Mbar, Hbar, Gbar)
% Prepare quadrature nodes / weights 

% a) 40 nodes
K = 40;    J = K;      N = K + J; % N / 2 is the no. of quadrature nodes for each rule
nodes_40 = 1i*logspace(-1, 4, N/2)'; % Halve since we will add complex conjugates
% Interweave left/right points
nodesl_40 = nodes_40(1:2:end); % For Q
nodesr_40 = nodes_40(2:2:end); % For P
% Close left/right points under conjugation
nodesl_40 = ([nodesl_40; conj(flipud(nodesl_40))]);     nodesr_40 = ([nodesr_40; conj(flipud(nodesr_40))]);    
% Quadrature weights according to composite Trapezoidal rule
weightsr_40 = [nodesr_40(2) - nodesr_40(1) ; nodesr_40(3:end) - nodesr_40(1:end-2) ; nodesr_40(end) - nodesr_40(end-1)] / 2;
weightsl_40 = [nodesl_40(2) - nodesl_40(1) ; nodesl_40(3:end) - nodesl_40(1:end-2) ; nodesl_40(end) - nodesl_40(end-1)] / 2;
weightsr_40 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsr_40));    weightsl_40 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsl_40)); 


GQBT_Engine_40 = GeneralizedQuadBTReductor(sampler, nodesl_40, nodesr_40, weightsl_40, weightsr_40);

% Some sanity checks
Utilde_40 = sampler.right_sqrt_factor(nodesr_40, weightsr_40);
Ltilde_40 = sampler.left_sqrt_factor(nodesl_40, weightsl_40);

% Check quadrature error
disp("Implicit quadrature error in P; 80 nodes:")
norm(Utilde_40 * Utilde_40' - sampler.P, 2)/norm(sampler.P)
disp("Implict quadrarture error in Q; 80 nodes:")
norm(Ltilde_40 * Ltilde_40' - sampler.Q, 2)/norm(sampler.Q)

% b) 80 nodes
K = 80;    J = K;      N = K + J; % N / 2 is the no. of quadrature nodes for each rule
nodes_80 = 1i*logspace(-1, 4, N/2)'; % Halve since we will add complex conjugates
% Interweave left/right points
nodesl_80 = nodes_80(1:2:end); % For Q
nodesr_80 = nodes_80(2:2:end); % For P
% Close left/right points under conjugation
nodesl_80 = ([nodesl_80; conj(flipud(nodesl_80))]);     nodesr_80 = ([nodesr_80; conj(flipud(nodesr_80))]);    
% Quadrature weights according to composite Trapezoidal rule
weightsr_80 = [nodesr_80(2) - nodesr_80(1) ; nodesr_80(3:end) - nodesr_80(1:end-2) ; nodesr_80(end) - nodesr_80(end-1)] / 2;
weightsl_80 = [nodesl_80(2) - nodesl_80(1) ; nodesl_80(3:end) - nodesl_80(1:end-2) ; nodesl_80(end) - nodesl_80(end-1)] / 2;
weightsr_80 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsr_80));    weightsl_80 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsl_80)); 


GQBT_Engine_80 = GeneralizedQuadBTReductor(sampler, nodesl_80, nodesr_80, weightsl_80, weightsr_80);

% Some sanity checks
Utilde_80 = sampler.right_sqrt_factor(nodesr_80, weightsr_80);
Ltilde_80 = sampler.left_sqrt_factor(nodesl_80, weightsl_80);

% Check quadrature error
disp("Implicit quadrature error in P; 80 nodes:")
norm(Utilde_80 * Utilde_80' - sampler.P, 2)/norm(sampler.P)
disp("Implict quadrarture error in Q; 80 nodes:")
norm(Ltilde_80 * Ltilde_80' - sampler.Q, 2)/norm(sampler.Q)

% Does Loewner build work?
Lbar = GQBT_Engine_80.Lbar;    Mbar = GQBT_Engine_80.Mbar;    Hbar = GQBT_Engine_80.Hbar;    Gbar = GQBT_Engine_80.Gbar; 

disp("Does Loewner build work? Lbar:")
norm(Ltilde_80' * Utilde_80 - Lbar, 2)
disp("Does Loewner build work? Mbar:")
norm(Ltilde_80' * A * Utilde_80 - Mbar, 2)
disp("Does Loewner build work? Gbar:")
norm(Ltilde_80' * B - Hbar, 2)
disp("Does Loewner build work? Hbar:")
norm(C * Utilde_80 - Gbar, 2)

% c) 160 nodes
K = 160;    J = K;      N = K + J; % N / 2 is the no. of quadrature nodes for each rule
nodes_160 = 1i*logspace(-1, 4, N/2)'; % Halve since we will add complex conjugates
% Interweave left/right points
nodesl_160 = nodes_160(1:2:end); % For Q
nodesr_160 = nodes_160(2:2:end); % For P
% Close left/right points under conjugation
nodesl_160 = ([nodesl_160; conj(flipud(nodesl_160))]);     nodesr_160 = ([nodesr_160; conj(flipud(nodesr_160))]);    
% Quadrature weights according to composite Trapezoidal rule
weightsr_160 = [nodesr_160(2) - nodesr_160(1) ; nodesr_160(3:end) - nodesr_160(1:end-2) ; nodesr_160(end) - nodesr_160(end-1)] / 2;
weightsl_160 = [nodesl_160(2) - nodesl_160(1) ; nodesl_160(3:end) - nodesl_160(1:end-2) ; nodesl_160(end) - nodesl_160(end-1)] / 2;
weightsr_160 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsr_160));    weightsl_160 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsl_160)); 


GQBT_Engine_160 = GeneralizedQuadBTReductor(sampler, nodesl_160, nodesr_160, weightsl_160, weightsr_160);

% Some sanity checks
Utilde_160 = sampler.right_sqrt_factor(nodesr_160, weightsr_160);
Ltilde_160 = sampler.left_sqrt_factor(nodesl_160, weightsl_160);

% Check quadrature error
disp("Implicit quadrature error in P; 160 nodes:")
norm(Utilde_160 * Utilde_160' - sampler.P, 2)/norm(sampler.P)
disp("Implict quadrarture error in Q; 160 nodes:")
norm(Ltilde_160 * Ltilde_160' - sampler.Q, 2)/norm(sampler.Q)


%% 3. Let's compare the approximate hsvs against the true ones
[~, ~, ~] = GQBT_Engine_40.svd_from_data;
hsvbar_40 = GQBT_Engine_40.hsvbar;
[~, ~, ~] = GQBT_Engine_80.svd_from_data;
hsvbar_80 = GQBT_Engine_80.hsvbar;
[~, ~, ~] = GQBT_Engine_160.svd_from_data;
hsvbar_160 = GQBT_Engine_160.hsvbar;
% Compute actual square-root factors (add some noise; rounding errors make
% these not positive definite)
% U = chol(sampler.P + (10e-16 * eye(n, n)));    L = chol(sampler.Q + (10e-13 * eye(n, n)));

r = 34;
% 
opts = ml_morlabopts('ml_ct_ss_bt');
opts.Order = r;
opts.OrderComputation = 'Order';
[Ar, br, cr, dr, output_opts] = ml_ct_ss_prbt(A, B, C, D, opts);

pr_hsvs = output_opts.Hsv;
max_x = r;

%% 3b) Plot

% New style (yoinked form Victor)

ColMat = zeros(5,3);

%ColMat(1,:) = [1 0.6 0.4];
ColMat(1,:) = [ 0.8500    0.3250    0.0980];
ColMat(2,:) = [0.3010    0.7450    0.9330];
ColMat(3,:) = [  0.9290    0.6940    0.1250];
%ColMat(3,:) = [1 0.4 0.6];
ColMat(4,:) = [0.4660    0.6740    0.1880];
ColMat(5,:) = [0.4940    0.1840    0.5560];

% f=figure;
% f.Position = [476 445 700 280];
% Make aspect ration `golden'
figure
golden_ratio = (sqrt(5)+1)/2;
axes('position', [.125 .15 .75 golden_ratio-1])
subplot(2,1,1)
semilogy(1:max_x, pr_hsvs(1:max_x), 'o','color',ColMat(1,:), LineWidth=1.5,MarkerSize=10)
hold on
semilogy(1:max_x, hsvbar_40(1:max_x), 'x', 'color',ColMat(3,:), LineWidth=1.5)
semilogy(1:max_x, hsvbar_80(1:max_x), '+', 'color',ColMat(4,:), LineWidth=1.5)
semilogy(1:max_x, hsvbar_160(1:max_x), '*', 'color',ColMat(2,:), LineWidth=1.5)
grid on
lgd = legend('True', 'Approx $(N = 40)$', 'Approx $(N = 80)$', 'Approx $(N = 160)$', 'interpreter','latex');
fontsize(lgd,10,'points')
set(lgd, 'FontName','Arial')
title('Singular values', 'interpreter','latex', 'fontsize', 14)
% xlabel('$k$, index', 'interpreter','latex', 'fontsize', 14)


disp('Frobenius norm error of the approximate PR HSVs; 40 nodes')
norm(diag(hsvbar_40(1:max_x))-diag(pr_hsvs(1:max_x)), "fro")
disp('Frobenius norm error of the approximate PR HSVs; 80 nodes')
norm(diag(hsvbar_80(1:max_x))-diag(pr_hsvs(1:max_x)), "fro")
disp('Frobenius norm error of the approximate PR HSVs; 160 nodes')
norm(diag(hsvbar_160(1:max_x))-diag(pr_hsvs(1:max_x)), "fro")

%% 4. Now, reduction error
FOM = ss(A, B, C, D); % FOM
Drbar = D; % d unchanged, always
sysnorm = norm(FOM, 'inf');

% Allocate space
testcases = 10;
PRBT_errors = zeros(testcases,1);   
QPRBT_40_errors = zeros(testcases,1);    QPRBT_80_errors = zeros(testcases,1);
QPRBT_160_errors = zeros(testcases,1); 
for k = 1:testcases % orders to test
    r = 2*k; % reduction order
    [Arbar_40, Brbar_40, Crbar_40] = GQBT_Engine_40.reduce(r);
    [Arbar_80, Brbar_80, Crbar_80] = GQBT_Engine_80.reduce(r);
    [Arbar_160, Brbar_160, Crbar_160] = GQBT_Engine_160.reduce(r);
    opts = ml_morlabopts('ml_ct_ss_bt');
    opts.Order = r;
    opts.OrderComputation = 'Order';
    [Ar, Br, Cr, Dr, output_opts] = ml_ct_ss_prbt(A, B, C, D, opts);
        
    PRBT_ROM = ss(Ar, Br, Cr, Dr);
    QPRBT_ROM_40 = ss(Arbar_40, Brbar_40, Crbar_40, Drbar);
    QPRBT_ROM_80 = ss(Arbar_80, Brbar_80, Crbar_80, Drbar);
    QPRBT_ROM_160 = ss(Arbar_160, Brbar_160, Crbar_160, Drbar);

    PRBT_errors(k, 1) = norm(FOM-PRBT_ROM, 'inf')/sysnorm;
%     fprintf("Error in quadprbt")
%     norm(sys-sysrbar, 'inf')
    QPRBT_40_errors(k, 1) = norm(FOM-QPRBT_ROM_40, 'inf')/sysnorm;
    QPRBT_80_errors(k, 1) = norm(FOM-QPRBT_ROM_80, 'inf')/sysnorm;
    QPRBT_160_errors(k, 1) = norm(FOM-QPRBT_ROM_160, 'inf')/sysnorm;
end



%% 4b) Plot

% f=figure;
% f.Position = [476 445 700 280];
% Make aspect ration `golden'
% golden_ratio = (sqrt(5)+1)/2;
% axes('position', [.125 .15 .75 golden_ratio-1])
subplot(2,1,2)
semilogy(2:2:2*testcases, PRBT_errors,'ms','color',ColMat(1,:),'markersize',15,LineWidth=1.5);hold on;
semilogy(2:2:2*testcases, QPRBT_40_errors,'-.g<','color',ColMat(3,:),LineWidth=1.5);
semilogy(2:2:2*testcases, QPRBT_80_errors,'--mo','color', ColMat(4,:),LineWidth=1.5);
semilogy(2:2:2*testcases, QPRBT_160_errors,'-.r','color',ColMat(2,:),LineWidth=1.5);

lgd = legend('PRBT', 'QPRBT $(N = 40)$', 'QPRBT $(N = 80)$', 'QPRBT $(N = 160)$', 'interpreter','latex');

xlabel('$r$, reduction order', 'interpreter','latex', 'fontsize', 14)
title('Relative $\mathcal{H}_{\infty}$ error', 'interpreter','latex', 'fontsize', 14)
