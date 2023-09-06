% Make sure you have the necessary directory added to your MATLAB path
addpath('/Users/seanr/Desktop/gen_quadbt/gen_qbt_matlab')
addpath('/Users/seanr/Desktop/gen_quadbt/gen_qbt_matlab/benchmarks/')
clear all

%% 1. Load benchmark for testing 
load('iss.mat')
% Convert spare -> Full
n = 270;
A = full(A);    B = full(B);    C = full(C);
% Include some nontrivial feedthrough
eps = 1e-2;
D = eps * eye(3, 3);
% Need to normalize FOM s.t. it is BR
FOM_ = ss(A, B, C, D);
gamma = norm(FOM_, 'inf');  gamma = gamma + .5;
D = D / gamma;  C = C / sqrt(gamma);    B = B / sqrt(gamma);
normalized_FOM_ = ss(A, B, C, D);
disp('Sanity check; is the FOM normalized?')
norm(normalized_FOM_, 'inf')

% Sampler class
sampler = QuadBRBTSampler(A, B, C, D);

%% 2. Instantiate reductor + build Loewner quadruple (Lbar, Mbar, Hbar, Gbar)
% Prepare quadrature nodes / weights 

% a) 200 nodes
K = 200;    J = K;      N = K + J; % N / 2 is the no. of quadrature nodes for each rule
nodes_200 = 1i*logspace(-1, 2, N/2)'; % Halve since we will add complex conjugates
% Interweave left/right points
nodesl_200 = nodes_200(1:2:end); % For Q
nodesr_200 = nodes_200(2:2:end); % For P
% Close left/right points under conjugation
nodesl_200 = ([nodesl_200; conj(flipud(nodesl_200))]);     nodesr_200 = ([nodesr_200; conj(flipud(nodesr_200))]);    
% Quadrature weights according to composite Trapezoidal rule
weightsr_200 = [nodesr_200(2) - nodesr_200(1) ; nodesr_200(3:end) - nodesr_200(1:end-2) ; nodesr_200(end) - nodesr_200(end-1)] / 2;
weightsl_200 = [nodesl_200(2) - nodesl_200(1) ; nodesl_200(3:end) - nodesl_200(1:end-2) ; nodesl_200(end) - nodesl_200(end-1)] / 2;
weightsr_200 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsr_200));    weightsl_200 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsl_200)); 


GQBT_Engine_200 = GeneralizedQuadBTReductor(sampler, nodesl_200, nodesr_200, weightsl_200, weightsr_200);

% Some sanity checks
Utilde_200 = sampler.right_sqrt_factor(nodesr_200, weightsr_200);
Ltilde_200 = sampler.left_sqrt_factor(nodesl_200, weightsl_200);

% Check quadrature error
disp("Implicit quadrature error in P; 200 nodes:")
norm(Utilde_200 * Utilde_200' - sampler.P, 2)/norm(sampler.P)
disp("Implict quadrarture error in Q; 200 nodes:")
norm(Ltilde_200 * Ltilde_200' - sampler.Q, 2)/norm(sampler.Q)

% b) 400 nodes
K = 400;    J = K;      N = K + J; % N / 2 is the no. of quadrature nodes for each rule
nodes_400 = 1i*logspace(-1, 2, N/2)'; % Halve since we will add complex conjugates
% Interweave left/right points
nodesl_400 = nodes_400(1:2:end); % For Q
nodesr_400 = nodes_400(2:2:end); % For P
% Close left/right points under conjugation
nodesl_400 = ([nodesl_400; conj(flipud(nodesl_400))]);     nodesr_400 = ([nodesr_400; conj(flipud(nodesr_400))]);    
% Quadrature weights according to composite Trapezoidal rule
weightsr_400 = [nodesr_400(2) - nodesr_400(1) ; nodesr_400(3:end) - nodesr_400(1:end-2) ; nodesr_400(end) - nodesr_400(end-1)] / 2;
weightsl_400 = [nodesl_400(2) - nodesl_400(1) ; nodesl_400(3:end) - nodesl_400(1:end-2) ; nodesl_400(end) - nodesl_400(end-1)] / 2;
weightsr_400 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsr_400));    weightsl_400 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsl_400)); 


GQBT_Engine_400 = GeneralizedQuadBTReductor(sampler, nodesl_400, nodesr_400, weightsl_400, weightsr_400);

% Some sanity checks
Utilde_400 = sampler.right_sqrt_factor(nodesr_400, weightsr_400);
Ltilde_400 = sampler.left_sqrt_factor(nodesl_400, weightsl_400);

% Check quadrature error
disp("Implicit quadrature error in P; 400 nodes:")
norm(Utilde_400 * Utilde_400' - sampler.P, 2)/norm(sampler.P)
disp("Implict quadrarture error in Q; 400 nodes:")
norm(Ltilde_400 * Ltilde_400' - sampler.Q, 2)/norm(sampler.Q)

% Does Loewner build work?
Lbar = GQBT_Engine_400.Lbar;    Mbar = GQBT_Engine_400.Mbar;    Hbar = GQBT_Engine_400.Hbar;    Gbar = GQBT_Engine_400.Gbar; 

disp("Does Loewner build work? Lbar:")
norm(Ltilde_400' * Utilde_400 - Lbar, 2)
disp("Does Loewner build work? Mbar:")
norm(Ltilde_400' * A * Utilde_400 - Mbar, 2)
disp("Does Loewner build work? Gbar:")
norm(Ltilde_400' * B - Hbar, 2)
disp("Does Loewner build work? Hbar:")
norm(C * Utilde_400 - Gbar, 2)

% c) 800 nodes
K = 800;    J = K;      N = K + J; % N / 2 is the no. of quadrature nodes for each rule
nodes_800 = 1i*logspace(-1, 2, N/2)'; % Halve since we will add complex conjugates
% Interweave left/right points
nodesl_800 = nodes_800(1:2:end); % For Q
nodesr_800 = nodes_800(2:2:end); % For P
% Close left/right points under conjugation
nodesl_800 = ([nodesl_800; conj(flipud(nodesl_800))]);     nodesr_800 = ([nodesr_800; conj(flipud(nodesr_800))]);    
% Quadrature weights according to composite Trapezoidal rule
weightsr_800 = [nodesr_800(2) - nodesr_800(1) ; nodesr_800(3:end) - nodesr_800(1:end-2) ; nodesr_800(end) - nodesr_800(end-1)] / 2;
weightsl_800 = [nodesl_800(2) - nodesl_800(1) ; nodesl_800(3:end) - nodesl_800(1:end-2) ; nodesl_800(end) - nodesl_800(end-1)] / 2;
weightsr_800 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsr_800));    weightsl_800 = sqrt(1 / (2 * pi)) * sqrt(abs(weightsl_800)); 


GQBT_Engine_800 = GeneralizedQuadBTReductor(sampler, nodesl_800, nodesr_800, weightsl_800, weightsr_800);

% Some sanity checks
Utilde_800 = sampler.right_sqrt_factor(nodesr_800, weightsr_800);
Ltilde_800 = sampler.left_sqrt_factor(nodesl_800, weightsl_800);

% Check quadrature error
disp("Implicit quadrature error in P; 800 nodes:")
norm(Utilde_800 * Utilde_800' - sampler.P, 2)/norm(sampler.P)
disp("Implict quadrarture error in Q; 800 nodes:")
norm(Ltilde_800 * Ltilde_800' - sampler.Q, 2)/norm(sampler.Q)


%% 3. Let's compare the approximate hsvs against the true ones
[~, ~, ~] = GQBT_Engine_200.svd_from_data;
hsvbar_200 = GQBT_Engine_200.hsvbar;
[~, ~, ~] = GQBT_Engine_400.svd_from_data;
hsvbar_400 = GQBT_Engine_400.hsvbar;
[~, ~, ~] = GQBT_Engine_800.svd_from_data;
hsvbar_800 = GQBT_Engine_800.hsvbar;
% Compute actual square-root factors (add some noise; rounding errors make
% these not positive definite)
% U = chol(sampler.P + (10e-16 * eye(n, n)));    L = chol(sampler.Q + (10e-13 * eye(n, n)));
% stoch_hsvs = sqrt(eig(sampler.P * sampler.Q));

r = 270;
% 
opts = ml_morlabopts('ml_ct_ss_bt');
opts.Order = r;
opts.OrderComputation = 'Order';
[Ar, br, cr, dr, output_opts] = ml_ct_ss_brbt(A, B, C, D, opts);

br_hsvs = output_opts.Hsv;
max_x = 100;

figure
% Make aspect ration `golden'
golden_ratio = (sqrt(5)+1)/2;
axes('position', [.125 .15 .75 golden_ratio-1])
semilogy(1:max_x, br_hsvs(1:max_x), 'o', LineWidth=1.5,MarkerSize=10)
hold on
semilogy(1:max_x, hsvbar_200(1:max_x), 'x', LineWidth=1.5)
semilogy(1:max_x, hsvbar_400(1:max_x), '+', LineWidth=1.5)
semilogy(1:max_x, hsvbar_800(1:max_x), '*', LineWidth=1.5)
grid on
ylabel('$\sqrt{\lambda_k(\mathbf{P}\mathbf{Q}_{\mathcal{W}})}$', 'interpreter','latex')
xlabel('$k$', 'interpreter','latex')
legend('True', 'Approximate, $N = 200$', 'Approximate, $N = 400$', 'Approximate, $N = 800$', 'interpreter','latex')

disp('Frobenius norm error of the approximate BRHSVs; 200 nodes')
norm(diag(hsvbar_200(1:max_x))-diag(br_hsvs(1:max_x)), "fro")
disp('Frobenius norm error of the approximate BRHSVs; 400 nodes')
norm(diag(hsvbar_400(1:max_x))-diag(br_hsvs(1:max_x)), "fro")
disp('Frobenius norm error of the approximate BRHSVs; 800 nodes')
norm(diag(hsvbar_800(1:max_x))-diag(br_hsvs(1:max_x)), "fro")

%% 4. Now, reduction error
FOM = ss(A, B, C, D); % FOM
Drbar = D; % d unchanged, always
sysnorm = norm(FOM, 'inf');

% Allocate space
testcases = 7;
BRBT_errors = zeros(testcases,1);   
QBRBT_200_errors = zeros(testcases,1);    QBRBT_400_errors = zeros(testcases,1);
QBRBT_800_errors = zeros(testcases,1); 
for k = 1:testcases % orders to test
    r = 2*k; % reduction order
    [Arbar_200, Brbar_200, Crbar_200] = GQBT_Engine_200.reduce(r);
    [Arbar_400, Brbar_400, Crbar_400] = GQBT_Engine_400.reduce(r);
    [Arbar_800, Brbar_800, Crbar_800] = GQBT_Engine_800.reduce(r);
    opts = ml_morlabopts('ml_ct_ss_bt');
    opts.Order = r;
    opts.OrderComputation = 'Order';
    [Ar, Br, Cr, Dr, output_opts] = ml_ct_ss_brbt(A, B, C, D, opts);
        
    BST_ROM = ss(Ar, Br, Cr, Dr);
    QBRBT_ROM_200 = ss(Arbar_200, Brbar_200, Crbar_200, Drbar);
    QBRBT_ROM_400 = ss(Arbar_400, Brbar_400, Crbar_400, Drbar);
    QBRBT_ROM_800 = ss(Arbar_800, Brbar_800, Crbar_800, Drbar);

    BRBT_errors(k, 1) = norm(FOM-BST_ROM, 'inf')/sysnorm;
%     fprintf("Error in quadprbt")
%     norm(sys-sysrbar, 'inf')
    QBRBT_200_errors(k, 1) = norm(FOM-QBRBT_ROM_200, 'inf')/sysnorm;
    QBRBT_400_errors(k, 1) = norm(FOM-QBRBT_ROM_400, 'inf')/sysnorm;
    QBRBT_800_errors(k, 1) = norm(FOM-QBRBT_ROM_800, 'inf')/sysnorm;
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

semilogy(2:2:2*testcases, BRBT_errors,'ms','color',ColMat(1,:),'markersize',15,LineWidth=1.5);hold on;
semilogy(2:2:2*testcases, QBRBT_200_errors,'-.g<','color',ColMat(3,:),LineWidth=1.5);
semilogy(2:2:2*testcases, QBRBT_400_errors,'--mo','color', ColMat(4,:),LineWidth=1.5);
semilogy(2:2:2*testcases, QBRBT_800_errors,'-.r','color',ColMat(2,:),LineWidth=1.5);

legend('PGRoM BRBT', 'QBRBT, $N = 200$', 'QBRBT, $N = 400$', 'QBRBT, $N = 800$', 'interpreter','latex')

% semilogy([2:2:2*testcases], BST_errors, '-s', Markersize = 10, LineWidth=1.5)
% hold on
% % grid on
% semilogy([2:2:2*testcases], QBST_20_errors, '-x', Linewidth=1.5)
% set(gca,'fontsize',12)
xlabel('$r$, reduction order', 'interpreter','latex')
ylabel('$\|\mathcal{G}-\mathcal{G}_r\|_{\mathcal{H}_\infty}/\|\mathcal{G}\|_{\mathcal{H}_\infty}$', 'interpreter','latex')