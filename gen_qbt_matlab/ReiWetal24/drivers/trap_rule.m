function [nodesl, weightsl, nodesr, weightsr] = trap_rule(exp_limits, N, interweave)
% Prepare quadrature weights and nodes according to the composite
% Trapezoidal rule.

% For use in GenQuadBT @cite: [R., Gugercin, Gosea, 24']

%
% Paramters
% ---------
% @param limits:
%   Limits of integration as (2, 1) array; left/right quadrature nodes
%   will be logarithmically points contained in the interval 
%   -i[10^b, 10^a] U i[10^a, 10^b],
%   where a = limits(1), b = limits(2).

% @param N:
%   Number of quadrature nodes used in each rule. Assumed to be even

% @param interweave:
%   Bool variable; do we interlace the left/right quadrature nodes along
%   iR? If not, same nodes used for left and right rules.

% Outputs
% -------
% @param nodesl:
%   Left quadrature nodes (used implicitly in approximating `relevant'
%   observability Gramian, Qy) as (N, 1) array.
%   Closed under complex conjugation.
% @param weightsl:
%   Left quadrature weights as (N, 1) array.

% @param nodesr:
%   Right quadrature nodes (used implicitly in approximating `relevant'
%   controllability Gramian, Px) as (N, 1) array.
%   Closed under complex conjugation.

% @param weightsr:
%   Right quadrature weights) as (N, 1) array.

%%
% Set default input parameters
if nargin < 1
    exp_limits = [-1, 3];
end
if nargin < 2
    N = 100;
end
if nargin < 3
    interweave = true;
end

if mod(N, 2) == 1
    fprintf('N is assumed to be even; setting N = N + 1')
    N = N + 1;
end

if interweave % If interweaving quadrature nodes
    nodes = 1i*logspace(exp_limits(1), exp_limits(2), N)';
    % Interweave left/right points
    % At this point, nodesl and nodesr have length N/2
    nodesl = nodes(1:2:end);    nodesr = nodes(2:2:end); 
    % Closeing under conj gives N nodes per rule
    nodesl = ([nodesl; conj(flipud(nodesl))]);     nodesr = ([nodesr; conj(flipud(nodesr))]);
else % Same nodes for each rule
    nodes = 1i*logspace(exp_limits(1), exp_limits(2), N / 2)';
    % Close under conjugation
    nodes = ([nodes; conj(flipud(nodes))]);  
    % Set return values
    nodesl = nodes; nodesr = nodes;
    
end
% Compute weights according to composite Trap rule
weightsr = [nodesr(2) - nodesr(1) ; nodesr(3:end) - nodesr(1:end-2) ; nodesr(end) - nodesr(end-1)] / 2;
weightsr = sqrt(1 / (2 * pi)) * sqrt(abs(weightsr));   
weightsl = [nodesl(2) - nodesl(1) ; nodesl(3:end) - nodesl(1:end-2) ; nodesl(end) - nodesl(end-1)] / 2; 
weightsl = sqrt(1 / (2 * pi)) * sqrt(abs(weightsl)); 
end