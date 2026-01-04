function [objFunc] = rayleigh_damping_obj(params, ...
                                          leftPoints,  rightPoints, ...
                                          leftWeights, rightWeights, ...
                                          leftData,    rightData,    r)
%RAYLEIGH_DAMPING_OBJ Function to output minimization objective for the 
% minimization problem (5.3) when the reduced-order model is not given. 
%
% SYNTAX:
%    objFunc = rayleigh_damping_obj(params, ...
%                                   leftPoints,  rightPoints, ...
%                                   leftWeights, rightWeights, ...
%                                   leftData,    rightData,    r)
% 
% DESCRIPTION:
%   This is a function to evaluate the minimization objective for the 
%   minimization problem (5.3) in the companion paper, to be used in 
%   conjunction with soQuadpvBT when the Rayleigh damping coefficients 
%   are unknown. The reduced-order model is recomputed via soQuadpvBT at
%   every evaluation of the objective function. 
%
% INPUTS:
%   params       - (alpha, beta) 2 x 1 dimensional vector containing
%                  minimization variables
%   leftPoints   - nNodes x 1 dimensional vector containing left 
%                  evaluation points
%   rightPoints  - nNodes x 1 dimensional vector containing right 
%                  evaluation points
%   leftWeights  - nNodes x 1 dimensional vector containing left weights 
%                  for diagonal scaling
%   rightWeights - nNodes x 1 dimensional vector containing right weights 
%                  for diagonal scaling
%   leftData     - p x m x nNodes dimensional array containing left 
%                  transfer function data of so system
%   rightData    - p x m x nNodes dimensional array containing right 
%                  transfer function data of so system
%   r            - reduced model order
% OUTPUTS:
%   objFunc - value of the objective function of the minimization problem 
%             (5.3) in the companion paper, evaluated at params

%
% This file is part of the archive Code, Data and Results for Numerical 
% Experiments in "Data-driven balanced truncation for second-order 
% systems with generalized proportional damping"
% Copyright (c) 2025 Sean Reiter, Steffen W. R. Werner
% All rights reserved.
% License: BSD 2-Clause license (see COPYING)
%
% Last editied: 1/4/2026
%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK INPUTS.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract relevant info from inputs. 
assert(length(leftPoints) == length(rightPoints), ...
    'It is assumed that there are an equal number of left and right data points!');
nNodes = length(leftPoints);
[p, m] = size(leftData(:, :, 1));
Ir     = eye(r, r);

% Given damping coefficients.
alpha = params(1);
beta  = params(2);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBJECTIVE FUNCTION.                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For given parameter values (alpha, beta), reassemble ROM using
% soQuadpvBT.

% Loewner matrices.
[Mbar_soQuadBT, ~, Kbar_soQuadBT, Bbar_soQuadBT, CpBar_soQuadBT] = ...
    so_loewner_factory(leftPoints, rightPoints, leftWeights, rightWeights, leftData, ...
                       rightData, 'Rayleigh', [alpha, beta], 'Position');

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

% Reductor.
[Z_soQuadBT, S_soQuadBT, Y_soQuadBT] = svd(Mbar_soQuadBT);

% Reduced model matrices.
Kr  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Kbar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
Cpr = CpBar_soQuadBT*(Y_soQuadBT(:, 1:r)*S_soQuadBT(1:r, 1:r)^(-1/2));
Br  = (S_soQuadBT(1:r, 1:r)^(-1/2)*Z_soQuadBT(:, 1:r)')*Bbar_soQuadBT;

% Using computed ROM, evaluate objective (5.3).
% Scalar-valued objective.
objFunc  = 0;
mismatch = zeros(p, m, 2*nNodes); % Malloc for error.

% Loop to evaluate objective function.
for k = 1:nNodes
    % tmp var; mismatch between data and reduced-order transfer
    % function at k-th node.
    mismatch(:, :, k) = leftData(:, :, k) - (Cpr)*(((leftPoints(k)^2 ...
                        + leftPoints(k)*alpha)*Ir + (1 + leftPoints(k)*beta)*Kr)\Br);
    objFunc           = objFunc + norm(mismatch(:, :, k), 'fro')^2;
end

i = 1;
for k = nNodes+1:2*nNodes
    % tmp var; mismatch between data and reduced-order transfer
    % function at k-th node.
    mismatch(:, :, k) = rightData(:, :, i) - (Cpr)*(((rightPoints(i)^2 ...
                        + rightPoints(i)*alpha)*Ir + (1 + rightPoints(i)*beta)*Kr)\Br);
    objFunc           = objFunc + norm(mismatch(:, :, k), 'fro')^2;
    i = i + 1;
end
end