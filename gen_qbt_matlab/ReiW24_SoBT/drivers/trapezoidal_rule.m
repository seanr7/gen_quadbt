function [nodesLeft, weightsLeft, nodesRight, weightsRight] = ...
    trapezoidal_rule(expLimits, numNodes, interweave)
%TRAPEZOIDALRULE Exponential trapeozidal rule.
%
% SYNTAX:
%   [nodesLeft, weightsLeft, nodesRight, weightsRight] = 
%       TrapezoidalRule(expLimits, numNodes, interweave)
%   [nodesLeft, weightsLeft, nodesRight, weightsRight] = 
%       TrapezoidalRule(expLimits, numNodes)
%
% DESCRIPTION:
%   Function to prepare quadrature nodes and weights for the exponential
%   trapezoidal rule applied to a pair of system Gramians formulated as
%   integrals along the imaginary axis.
%   The nodes are contained in the union of the intervals i[1e+a, 1e+b],
%   and -i[1e+a, 1e+b] where a = expLimits(0), b = expLimits(1).
%
% INPUT:
%   expLimits   - 2 x 1 matrix containing exponential limits of the 
%                 quadrature rule
%   numNodes    - number of quadrature nodes, assumed N = 2*K
%   interwewave - boolean; are the nodes interweaved among eachother? 
%
% OUTPUT: 
%   nodesLeft    - `left' quadrature nodes, used in approximation of the
%                  observability system Gramian
%   weightsLeft  - `left' quadrature weights
%   nodesRight   - `right' quadrature nodes, used in approximation of the
%                  controllability system Gramian
%   weightsRight - `right' quadrature weights
%

%
% This file is part of the archive Code and Results for Numerical 
% Experiments in "..."
% Copyright (c) 2024 Sean Reiter, Steffen W. R. Werner
% All rights reserved.
% License: BSD 2-Clause license (see COPYING)
%
% Virginia Tech, USA
% Last editied: 7/16/2024
%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK INPUTS.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 3
    % Default is to interweave.
    interweave = true;
end

if mod(numNodes, 2) == 1
    fprintf(1, 'Number of quadrature nodes must be even! Setting numNodes = numNodes + 1.\n')
    fprintf(1, '-------------------------------------------------------------------------\n')
    numNodes = numNodes + 1;
end

% Endpoints of exponential trapeozidal rule.
a = expLimits(1);   b = expLimits(2);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREPARE QUADRATURE RULES.                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if interweave 
    % Compute nodes.
    omega      = 1i*(logspace(a, b, numNodes)');
    nodesLeft  = omega(1:2:end);    
    nodesRight = omega(2:2:end); 
    % Close left and right nodes under complex conjugation.
    nodesLeft  = ([nodesLeft; conj(flipud(nodesLeft))]);     
    nodesRight = ([nodesRight; conj(flipud(nodesRight))]);
else 
    omega      = 1i*logspace(a, b, numNodes/2)';
    % Close left and right nodes under complex conjugation.
    omega      = ([omega; conj(flipud(omega))]);  
    nodesLeft  = omega; 
    nodesRight = omega; 
end

% Compute weights according to the exponential trapezoidal rule.
weightsRight = [nodesRight(2) - nodesRight(1); nodesRight(3:end) - nodesRight(1:end-2); ...
    nodesRight(end) - nodesRight(end-1)]./2;
weightsRight = sqrt(1/(2*pi))*sqrt(abs(weightsRight));   
weightsLeft  = [nodesLeft(2) - nodesLeft(1); nodesLeft(3:end) - nodesLeft(1:end-2); ...
    nodesLeft(end) - nodesLeft(end-1)]./2; 
weightsLeft  = sqrt(1/(2*pi))*sqrt(abs(weightsLeft)); 

end