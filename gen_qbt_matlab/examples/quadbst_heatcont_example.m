% Make sure you have the necessary directory added to your MATLAB path
addpath('/Users/seanr/Desktop/gen_quadbt/gen_qbt_matlab')
clear all

%% 1. Load benchmark for testing 
load('heat-cont.mat')
% Convert spare -> Full
n = 200;
A = full(A);    B = full(B);    C = full(C);
% Include some nontrivial feedthrough
eps = 1e-2;
D = eps;
% Sampler class
sampler = QuadBSTSampler(A, B, C, D);

%% 2. Instantiate reductor + build Loewner quadruple (Lbar, Mbar, Hbar, Gbar)
% Prepare quadrature nodes / weights 
% a) 20 nodes
K = 20;    J = K;      N = K + J; % N / 2 is the no. of quadrature nodes for each rule
nodes_20 = 1i*logspace(-3, 3, N/2)'; % Halve since we will add complex conjugates
% Interweave left/right points
nodesl_20 = nodes_20(1:2:end); % For Q
nodesr_20 = nodes_20(2:2:end); % For P
% Close left/right points under conjugation
nodesl_20 = ([nodesl_20; conj(flipud(nodesl_20))]);     nodesr_20 = ([nodesr_20; conj(flipud(nodesr_20))]);    
% Quadrature weights according to composite Trapezoidal rule
weightsr_20 = [nodesr_20(2) - nodesr_20(1) ; nodesr_20(3:end) - nodesr_20(1:end-2) ; nodesr_20(end) - nodesr_20(end-1)] / 2;
weightsl_20 = [nodesl_20(2) - nodesl_20(1) ; nodesl_20(3:end) - nodesl_20(1:end-2) ; nodesl_20(end) - nodesl_20(end-1)] / 2;
weightsr_20 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsr_20));    weightsl_20 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsl_20)); 


GQBT_Engine_20 = GeneralizedQuadBTReductor(sampler, nodesl_20, nodesr_20, weightsl_20, weightsr_20);

% Some sanity checks
Utilde_20 = sampler.right_sqrt_factor(nodesr_20, weightsr_20);
Ltilde_20 = sampler.left_sqrt_factor(nodesl_20, weightsl_20);

% Check quadrature error
display("Implicit quadrature error in P; 20 nodes:")
norm(Utilde_20 * Utilde_20' - sampler.P, 2)/norm(sampler.P)
display("Implict quadrarture error in Q; 20 nodes:")
norm(Ltilde_20 * Ltilde_20' - sampler.Q, 2)/norm(sampler.Q)


% b) 40 nodes
K = 40;    J = K;      N = K + J; % N / 2 is the no. of quadrature nodes for each rule
nodes_40 = 1i*logspace(-3, 3, N/2)'; % Halve since we will add complex conjugates
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
display("Implicit quadrature error in P; 40 nodes:")
norm(Utilde_40 * Utilde_40' - sampler.P, 2)/norm(sampler.P)
display("Implict quadrarture error in Q; 40 nodes:")
norm(Ltilde_40 * Ltilde_40' - sampler.Q, 2)/norm(sampler.Q)

% c) 80 nodes
K = 80;    J = K;      N = K + J; % N / 2 is the no. of quadrature nodes for each rule
nodes_80 = 1i*logspace(-3, 3, N/2)'; % Halve since we will add complex conjugates
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
display("Implicit quadrature error in P; 80 nodes:")
norm(Utilde_80 * Utilde_80' - sampler.P, 2)/norm(sampler.P)
display("Implict quadrarture error in Q; 80 nodes:")
norm(Ltilde_80 * Ltilde_80' - sampler.Q, 2)/norm(sampler.Q)

% Does Loewner build work?
Lbar = GQBT_Engine_80.Lbar;    Mbar = GQBT_Engine_80.Mbar;    Hbar = GQBT_Engine_80.Hbar;    Gbar = GQBT_Engine_80.Gbar; 

display("Does Loewner build work? Lbar:")
norm(Ltilde_80' * Utilde_80 - Lbar, 2)
display("Does Loewner build work? Mbar:")
norm(Ltilde_80' * A * Utilde_80 - Mbar, 2)
display("Does Loewner build work? Gbar:")
norm(Ltilde_80' * B - Hbar, 2)
display("Does Loewner build work? Hbar:")
norm(C * Utilde_80 - Gbar, 2)

%% 3. Let's compare the approximate hsvs against the true ones
[Zbar_20, Sbar_20, Ybar_h_20] = GQBT_Engine_20.svd_from_data;
hsvbar_20 = GQBT_Engine_20.hsvbar;
[Zbar_40, Sbar_40, Ybar_h_40] = GQBT_Engine_40.svd_from_data;
hsvbar_40 = GQBT_Engine_40.hsvbar;
[Zbar_80, Sbar_80, Ybar_h_80] = GQBT_Engine_80.svd_from_data;
hsvbar_80 = GQBT_Engine_80.hsvbar;
% Compute actual square-root factors (add some noise; rounding errors make
% these not positive definite)
% U = chol(sampler.P + (10e-16 * eye(n, n)));    L = chol(sampler.Q + (10e-13 * eye(n, n)));
stoch_hsvs = sqrt(eig(sampler.P * sampler.Q));

r = 200;
% 
opts = ml_morlabopts('ml_ct_ss_bt');
opts.Order = r;
opts.OrderComputation = 'Order';
[Ar, br, cr, dr, output_opts] = ml_ct_ss_bst(A, B, C, D, opts);

stoch_hsvs = output_opts.Hsv;
max_x = 20;

figure
% Make aspect ration `golden'
golden_ratio = (sqrt(5)+1)/2;
axes('position', [.125 .15 .75 golden_ratio-1])
semilogy(1:max_x, stoch_hsvs(1:max_x), 'o', LineWidth=1.5,MarkerSize=10)
hold on
semilogy(1:max_x, hsvbar_20(1:max_x), '*', LineWidth=1.5)
semilogy(1:max_x, hsvbar_40(1:max_x), 'x', LineWidth=1.5)
semilogy(1:max_x, hsvbar_80(1:max_x), '+', LineWidth=1.5)
grid on
ylabel('$\sqrt{\lambda_k(\mathbf{P}\mathbf{Q}_{\mathcal{W}})}$', 'interpreter','latex')
xlabel('$k$', 'interpreter','latex')
legend('True', 'Approximate, $N = 20$', 'Approximate, $N = 40$', 'Approximate, $N = 80$', 'interpreter','latex')

display('Frobenius norm error of the approximate Stochastic HSVs; 20 nodes')
norm(diag(hsvbar_20(1:max_x))-diag(stoch_hsvs(1:max_x)), "fro")
display('Frobenius norm error of the approximate Stochastic HSVs; 40 nodes')
norm(diag(hsvbar_40(1:max_x))-diag(stoch_hsvs(1:max_x)), "fro")
display('Frobenius norm error of the approximate Stochastic HSVs; 80 nodes')
norm(diag(hsvbar_80(1:max_x))-diag(stoch_hsvs(1:max_x)), "fro")

%% 4. Now, reduction error
FOM = ss(A, B, C, D); % FOM
Drbar = D; % d unchanged, always
sysnorm = norm(FOM, 'inf');

% Allocate space
testcases = 7;
BST_errors = zeros(testcases,1);   QBST_20_errors = zeros(testcases,1); 
QBST_40_errors = zeros(testcases,1);    QBST_80_errors = zeros(testcases,1);
for k = 1:testcases % orders to test
    r = 2*k; % reduction order
    [Arbar_20, Brbar_20, Crbar_20] = GQBT_Engine_20.reduce(r);
    [Arbar_40, Brbar_40, Crbar_40] = GQBT_Engine_40.reduce(r);
    [Arbar_80, Brbar_80, Crbar_80] = GQBT_Engine_80.reduce(r);
    opts = ml_morlabopts('ml_ct_ss_bt');
    opts.Order = r;
    opts.OrderComputation = 'Order';
    [Ar, Br, Cr, Dr, output_opts] = ml_ct_ss_bst(A, B, C, D, opts);
        
    BST_ROM = ss(Ar, Br, Cr, Dr);
    QBST_ROM_20 = ss(Arbar_20, Brbar_20, Crbar_20, Drbar);
    QBST_ROM_40 = ss(Arbar_40, Brbar_40, Crbar_40, Drbar);
    QBST_ROM_80 = ss(Arbar_80, Brbar_80, Crbar_80, Drbar);

    BST_errors(k, 1) = norm(FOM-BST_ROM, 'inf')/sysnorm;
%     fprintf("Error in quadprbt")
%     norm(sys-sysrbar, 'inf')
    QBST_20_errors(k, 1) = norm(FOM-QBST_ROM_20, 'inf')/sysnorm;
    QBST_40_errors(k, 1) = norm(FOM-QBST_ROM_40, 'inf')/sysnorm;
    QBST_80_errors(k, 1) = norm(FOM-QBST_ROM_80, 'inf')/sysnorm;
end

% New style (yoinked form Victor)

ColMat = zeros(5,3);

%ColMat(1,:) = [1 0.6 0.4];
ColMat(1,:) = [ 0.8500    0.3250    0.0980];
ColMat(2,:) = [0.3010    0.7450    0.9330];
ColMat(3,:) = [  0.9290    0.6940    0.1250];
%ColMat(3,:) = [1 0.4 0.6];
ColMat(4,:) = [0.4660    0.6740    0.1880];
ColMat(5,:) = [0.4940    0.1840    0.5560];


figure
% Make aspect ration `golden'
golden_ratio = (sqrt(5)+1)/2;
axes('position', [.125 .15 .75 golden_ratio-1])

semilogy([2:2:2*testcases], BST_errors,'ms','color',ColMat(1,:),'markersize',15,LineWidth=1.5);hold on;
semilogy([2:2:2*testcases], QBST_20_errors,'-.r','color',ColMat(2,:),LineWidth=1.5);
semilogy([2:2:2*testcases], QBST_40_errors,'-.g<','color',ColMat(3,:),LineWidth=1.5);
semilogy([2:2:2*testcases], QBST_80_errors,'--mo','color', ColMat(4,:),LineWidth=1.5);

legend('PGRoM BST', 'QBST, $N = 20$', 'QBST, $N = 40$', 'QBST, $N = 80$', 'interpreter','latex')

% semilogy([2:2:2*testcases], BST_errors, '-s', Markersize = 10, LineWidth=1.5)
% hold on
% % grid on
% semilogy([2:2:2*testcases], QBST_20_errors, '-x', Linewidth=1.5)
% set(gca,'fontsize',12)
xlabel('$r$, reduction order', 'interpreter','latex')
ylabel('$\|\mathcal{G}-\mathcal{G}_r\|_{\mathcal{H}_\infty}/\|\mathcal{G}\|_{\mathcal{H}_\infty}$', 'interpreter','latex')