function [MBAR, DBAR, KBAR, BBAR, CBAR] = so_loewner_factory(leftPoints,  rightPoints, ...
                                                             leftWeights, rightWeights, ...
                                                             leftData,    rightData, ...
                                                             damping,     DParam)
%LOEWNER_FACTORY Factory function for (second-order) Loewner matrix 
% quintuples.
%
% SYNTAX:
%   [MBAR, DBAR, KBAR, BBAR, CBAR] = loewner_factory(leftPoints,  rightPoints, ...
%                                                    leftWeights, rightWeights, ...
%                                                    leftData,    rightData, ...
%                                                    damping,     DParam)
%
% DESCRIPTION:
%   This is a factory function to compute Loewner matrix quintuples
%   (MBAR, DBAR, KBAR, BBAR, CBAR) for use in second-order (so)
%   quadrature-based balanced truncation. 
%   Formula for the Loewner matrices depend on the damping model;
%   see [Reiter and Werner, 2024].
%
%   To compute matrices for the second-order Loewner framework presents in
%   "Data-Driven Identification of Rayleigh-Damped Second-Order Systems" 
%   [Pontes, Goyal, and Benner 2022], set leftWeights and rightWeights to
%   be vectors of all ones.
%
% INPUTS:
%   leftPoints   - nLeft x 1 dimensional vector containing left 
%                  evaluation points
%   rightPoints  - nRight x 1 dimensional vector containing right 
%                  evaluation points
%   leftWeights  - nLeft x 1 dimensional vector containing left weights 
%                  for diagonal scaling
%   rightWeights - nRight x 1 dimensional vector containing right weights 
%                  for diagonal scaling
%   leftData     - p x m x nLeft dimensional array containing left 
%                  transfer function data of so system
%   rightData    - p x m x nRight dimensional array containing right 
%                  transfer function data of so system
%   damping      - string indicating the underlying damping model; options 
%                  are 'Rayleigh', and 'Structural'
%   DParam       - relevant damping parameters:
%                  if strcmp(damping, 'Rayleigh') 
%                       DParam = [alpha; beta] where D = alpha*M + beta*K
%                  if strcmp(damping, 'Structural')
%                       DParam = eta, material damping coefficient
%
% OUTPUTS: 
%   MBAR - nLeft*p x nRight*m diagonally scaled Loewner matrix; 
%          corresponds to mass matrix M in so system realization 
%   DBAR - nLeft*p x nRight*m diagonally scaled shifted Loewner matrix;
%          corresponds to damping matrix D in so system realization 
%   KBAR - nLeft*p x nRight*m diagonally scaled shifted Loewner matrix;
%          corresponds to stiffness matrix K in so system realization 
%   BBAR - nLeft*p x m matrix of left data; in (3) corresponds to input 
%          matrix B in so system realization
%   CBAR - p x nRight*m matrix of right data in (4); corresponds to 
%          (position) output matrix C in so system realization
% 

%
% This file is part of the archive Code and Results for Numerical 
% Experiments in "..."
% Copyright (c) 2024 Sean Reiter, Steffen W. R. Werner
% All rights reserved.
% License: BSD 2-Clause license (see COPYING)
%
% Virginia Tech, USA
% Last editied: 8/12/2024
%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK INPUTS.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dimensions.
nLeft  = length(leftPoints);    
nRight = length(rightPoints);
[p, m] = size(leftData(:, :, 1));

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
MBAR = zeros(nLeft*p, nRight*m);
DBAR = zeros(nLeft*p, nRight*m);
KBAR = zeros(nLeft*p, nRight*m);
BBAR   = zeros(nLeft*p, m);
CBAR   = zeros(p, nRight*m);

% Calculation of HBAR, GBAR, same regardless of damping model.
for k = 1:nLeft
    BBAR((k - 1)*p + 1:k*p, :) = leftWeights(k)*leftData(:, :, k); 
end
for j = 1:nRight
    CBAR(:, (j - 1)*m + 1:j*m) = rightWeights(j)*rightData(:, :, j);  
end

% Outside of the scalar function evaluations n(s), d(s), and f(s), the 
% construction of LBAR_M and LBAR_K proceed identically for 'Rayleigh' and
% 'Structural' damping.

% Compute additional function evaluations arising in Loewner 
% construction (scalar functions, so quick).
denomLeft  = zeros(nLeft, 1);
denomRight = zeros(nRight, 1);
numLeft    = zeros(nLeft, 1);
numRight   = zeros(nRight, 1);
fracLeft   = zeros(nLeft, 1);
fracRight  = zeros(nRight, 1);

% Rayleigh damping.
if strcmp(damping, 'Rayleigh')
    for k = 1:nLeft
        denomLeft(k) = 1 + beta*leftPoints(k);
        numLeft(k)   = leftPoints(k)^2 + alpha*leftPoints(k);
        fracLeft(k)  = numLeft(k)/denomLeft(k);
    end
    for j = 1:nRight
        denomRight(j) = 1 + beta*rightPoints(j);
        numRight(j)   = rightPoints(j)^2 + alpha*rightPoints(j);
        fracRight(j)  = numRight(j)/denomRight(j); 
    end
end

% Structural damping.
if strcmp(damping, 'Structural')
    % Denominator is constant, but store a bunch of the same value to keep
    % the Loewner factory more compact.
    tmp = 1i*eta + 1;
    for k = 1:nLeft
        denomLeft(k) = tmp;
        numLeft(k)   = leftPoints(k)^2;
        fracLeft(k)  = numLeft(k)/denomLeft(k);
    end
    for j = 1:nRight
        denomRight(j) = tmp;
        numRight(j)   = rightPoints(j)^2;
        fracRight(j)  = numRight(j)/denomRight(j); 
    end
end

% Block entrywise calculation of LBAR_M, LBAR_K.
for k = 1:nLeft
    for j = 1:nRight
        tmpDenom = fracLeft(k) - fracRight(j);
        % For LBAR_M.
        MBAR((k - 1)*p + 1:k*p, (j - 1)*m + 1:j*m) = -leftWeights(k)*rightWeights(j) ...
            *((1/denomRight(j))*leftData(:, :, k) - (1/denomLeft(k))*rightData(:, :, j))./tmpDenom;

        % For LBAR_K.
        KBAR((k - 1)*p + 1:k*p, (j - 1)*m + 1:j*m) = leftWeights(k)*rightWeights(j) ...
            *((fracLeft(k)/denomRight(j))*leftData(:, :, k) - (fracRight(j)/denomLeft(k))*rightData(:, :, j))./tmpDenom;
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