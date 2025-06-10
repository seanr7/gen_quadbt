function [MBAR, DBAR, KBAR, BBAR, CBAR] = so_hermite_loewner_factory(points, ...
                                                                     weights, ...
                                                                     data, ...
                                                                     derivData, ...
                                                                     damping, ...
                                                                     DParam)
%SO_HERMITE_LOEWNER_FACTORY Factory function for (second-order) Hermite
% Loewner matrix quintuples.
%
% SYNTAX:
%   [MBAR, DBAR, KBAR, BBAR, CBAR] = so_hermite_loewner_factory(points, ...
%                                                               weights, ...
%                                                               data, ...
%                                                               derivData, ...
%                                                               damping, ...
%                                                               DParam)
%
%
% DESCRIPTION:
%   This is a factory function to compute Loewner matrix quintuples
%   (MBAR, DBAR, KBAR, BBAR, CBAR) for use in second-order (so)
%   quadrature-based balanced truncation. 
%   Formula for the Loewner matrices depend on the damping model;
%   see the companion paper [Reiter and Werner, 2025].
%
%   To compute matrices for the so Loewner framework proposed in
%   "Data-driven identification of Rayleigh-damped second-order systems" 
%   [Pontes, Goyal, and Benner 2022], set leftWeights and weights to
%   be vectors of all ones.
%
% INPUTS:
%   points    - nNodes x 1 dimensional vector containing left evaluation 
%               points
%   weights   - nNodes x 1 dimensional vector containing weights for 
%               diagonal scaling
%   data      - p x m x nNodes dimensional array containing transfer 
%               function data of so system
%   derivData - p x m x nNodes dimensional array containing derivatives of
%               transfer function data 
%   damping   - string indicating the underlying damping model; options 
%               are 'Rayleigh', and 'Structural'
%   DParam   - relevant damping parameters:
%                if strcmp(damping, 'Rayleigh') 
%                  DParam = [alpha; beta] where D = alpha*M + beta*K
%                if strcmp(damping, 'Structural')
%                  DParam = eta, material damping coefficient
%
% OUTPUTS: 
%   MBAR - nNodes*p x nNodes*m diagonally scaled Hermite Loewner matrix; 
%          corresponds to mass matrix M in so system realization 
%   DBAR - nNodes*p x nNodes*m matrix; corresponds to damping matrix D in 
%          so system realization 
%   KBAR - nNodes*p x nNodes*m diagonally scaled shifted Hermite Loewner
%          matrix; corresponds to stiffness matrix K in so system 
%          realization 
%   BBAR - nNodes*p x m matrix of left data; in (3) corresponds to input 
%          matrix B in so system realization
%   CBAR - p x nNodes*m matrix of right data in (4); corresponds to 
%          position output matrix Cp in so system realization
% 

%
% This file is part of the archive Code, Data and Results for Numerical 
% Experiments in "Data-driven balanced truncation for second-order 
% systems with generalized proportional damping"
% Copyright (c) 2025 Sean Reiter, Steffen W. R. Werner
% All rights reserved.
% License: BSD 2-Clause license (see COPYING)
%
% Virginia Tech, USA
% Last editied: 6/09/2025
%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK INPUTS.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dimensions.
[p, m] = size(data(:, :, 1));
nNodes = length(points);

% Assertions.
assert(strcmp(damping, 'Rayleigh') || strcmp(damping, 'Structural'), ...
    'Inputted damping model not supported!')

if strcmp(damping, 'Rayleigh')
    % Rayleigh damping coefficients.
    alpha = DParam(1);  beta = DParam(2);
end
if strcmp(damping, 'Structural')
    % Structural damping coefficient.
    eta = DParam;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOEWNER FACTORY.                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Space allocation.
MBAR = zeros(nNodes*p, nNodes*m);
DBAR = zeros(nNodes*p, nNodes*m);
KBAR = zeros(nNodes*p, nNodes*m);
BBAR = zeros(nNodes*p, m);
CBAR = zeros(p, nNodes*m);

% Calculation of BBAR, CBAR, same regardless of damping model.
for k = 1:nNodes
    BBAR((k - 1)*p + 1:k*p, :) = weights(k)*data(:, :, k); 
    CBAR(:, (k - 1)*m + 1:k*m) = weights(k)*data(:, :, k);  
end

% Outside of the scalar function evaluations n(s), d(s), and f(s), the 
% construction of MBAR and KBAR proceed identically for 'Rayleigh' and
% 'Structural' damping.

% Compute additional function evaluations arising in Loewner 
% construction (scalar functions, so quick).
denom      = zeros(nNodes, 1);
denomDeriv = zeros(nNodes, 1);
num        = zeros(nNodes, 1);
numDeriv   = zeros(nNodes, 1);
frac       = zeros(nNodes, 1);
fracDeriv  = zeros(nNodes, 1);

% Rayleigh damping.
if strcmp(damping, 'Rayleigh')
    % Derivative of denominator d'(s) is constant beta, but store a bunch 
    % of the same value to keep code clean. 
    for k = 1:nNodes
        denom(k)      = 1 + beta*points(k);
        denomDeriv(k) = beta;
        num(k)        = points(k)^2 + alpha*points(k);
        numDeriv(k)   = 2*points(k) + alpha;
        frac(k)       = num(k)/denom(k);
        fracDeriv(k)  = (denom(k)*numDeriv(k) - num(k)*denomDeriv(k))/denom(k)^2;
    end
end

% Structural damping.
if strcmp(damping, 'Structural')
    % Denominator d(s) is constant 1 + i*eta, but store a bunch of the same 
    % value to keep code clean. d'(s) = 0, so denomDeriv stays as zeros.
    tmp = 1i*eta + 1;
    for k = 1:nNodes
        denom(k)     = tmp;
        num(k)       = points(k)^2;
        numDeriv(k)  = 2*points(k);
        frac(k)      = num(k)/denom(k);
        fracDeriv(k) = denom(k)*numDeriv(k)/denom(k)^2; 
    end
end

% Block entrywise calculation of LBAR_M, LBAR_K.
for k = 1:nNodes
    for j = 1:nNodes
        if k ~= j
            % Off-diagonal entries (no derivatives).
            tmpDenom = frac(k) - frac(j);
            tmpMult  = ((weights(k)*weights(j))/(denom(k)*denom(j)));
            
            % For MBAR.
            MBAR((k - 1)*p + 1:k*p, (j - 1)*m + 1:j*m) = -tmpMult*(denom(k)*data(:, :, k) ...
                - denom(j)*data(:, :, j))./tmpDenom;
    
            % For KBAR.
            KBAR((k - 1)*p + 1:k*p, (j - 1)*m + 1:j*m) = tmpMult*(num(k)*data(:, :, k) ...
                - num(j)*data(:, :, j))./tmpDenom;
        else
            % Diagonal entries (derivatives).
            % For MBAR.
            MBAR((k - 1)*p + 1:k*p, (k - 1)*m + 1:k*m) = - (weights(k)/denom(k))^2 ...
                *(data(:, :, k)*denomDeriv(k) + derivData(:, :, k)*denom(k))/fracDeriv(k);

            % For KBAR.
            KBAR((k - 1)*p + 1:k*p, (k - 1)*m + 1:k*m) = (weights(k)/denom(k))^2 ...
                *(data(:, :, k)*numDeriv(k) + derivData(:, :, k)*num(k))/fracDeriv(k);
        end

    end
end

% Damping matrix.
if strcmp(damping, 'Rayleigh')
    DBAR = alpha*MBAR + beta*KBAR;
end
if strcmp(damping, 'Structural')
    DBAR = 1i*eta*KBAR;
end

end