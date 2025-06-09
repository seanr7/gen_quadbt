function [objFunc, grads] = rayleigh_damping_obj(params, nodes, tfEvals, Kr, ...
                                                 Br,     Cpr,   Cvr)
%RAYLEIGH_DAMPING_OBJ Function to output minimization objective and
%   corresponding gradients for the minimization problem (5.3).
%
% SYNTAX:
%   [objFunc, grads] = rayleigh_damping_obj(params, nodes, tfEvals, Kr, ...
%                                           Br,     Cpr,   Cvr)
%   objFunc          = rayleigh_damping_obj(params, nodes, tfEvals, Kr, ...
%                                           Br,     Cpr,   Cvr)
%
%
% DESCRIPTION:
%   This is a function to evaluate the minimization objective and
%   corresponding gradients for the minimization problem (5.3) in the
%   companion paper, to be used in conjunction with soQuadpvBT.
%
% INPUTS:
%   params  - (alpha, beta) 2 x 1 dimensional vector containing
%             minimization variables
%   tfEvals - p x m x nNodes dimensional array containing data in the
%             objective function
%   Kr      - r x r (reduced) stiffness matrix
%   Br      - r x m (reduced) input matrix
%   Cpr     - p x r (reduced) position-output matrix
%   Cvr     - p x r (reduced) velocity-output matrix
%
% OUTPUTS: 
%   objFunc - value of the objective function of the minimization problem 
%             (5.3) in the companion paper, evaluated at params
%   grads   - gradients of the objective function of the minimization 
%             problem (5.3) in the companion paper, evaluated at params
% 

%
% This file is part of the archive Code and Results for Numerical 
% Experiments in "Data-driven balanced truncation for second-order systems
% with generalized proportional damping"
% Copyright (c) 2025 Sean Reiter, Steffen W. R. Werner
% All rights reserved.
% License: BSD 2-Clause license (see COPYING)
%
% Virginia Tech, USA
% Last editied: 5/28/2025
%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK INPUTS.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract relevant info from inputs. 
nNodes = length(nodes);
[p, r] = size(Cpr);    
[~, m] = size(Br);
Ir     = eye(r, r);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBJECTIVE FUNCTION.                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Scalar-valued objective.
objFunc  = 0;
mismatch = zeros(p, m, nNodes); % Malloc for error.

% Loop to evaluate objective function.
for k = 1:nNodes
    % tmp var; mismatch between data and reduced-order transfer
    % function at k-th node.
    mismatch(:, :, k) = tfEvals(:, :, k) - (Cpr + nodes(k)*Cvr)*(((nodes(k)^2 ...
                        + nodes(k)*params(1))*Ir + (1 + nodes(k)*params(2))*Kr)\Br);
    objFunc           = objFunc + norm(mismatch(:, :, k), 'fro')^2;
end

if nargout > 1
    grads = zeros(1, 2); 
    for k = 1:nNodes
        % tmp var; mismatch between data and reduced-order transfer
        % function at k-th node.
        
        % Gradient with respect to alpha (params(1)).
        tmpSolve = ((nodes(k)^2 + nodes(k)*params(1))*Ir + (1 + nodes(k)*params(2))*Kr)\Ir;
        grads(1) = grads(1) + 2*real(nodes(k)*trace((tmpSolve*Br)*mismatch(:, :, k)'*(Cpr + nodes(k)*Cvr)*tmpSolve));

        % Gradient with respect to beta (params(2)).
        grads(2) = grads(2) + 2*real(nodes(k)*trace(Kr*(tmpSolve*Br)*mismatch(:, :, k)'*(Cpr + nodes(k)*Cvr)*tmpSolve));
    end
end
end