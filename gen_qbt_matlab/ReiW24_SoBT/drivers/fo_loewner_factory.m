function [EBAR, ABAR, BBAR, CBAR] = fo_loewner_factory(leftPoints,  rightPoints, ...
                                                       leftWeights, rightWeights, ...
                                                       leftSamples, rightSamples)
%FO_LOEWNER_FACTORY Factory function for (first-order) Loewner matrix 
% quadruples.
%
% SYNTAX:
%   [EBAR, ABAR, BBAR, CBAR] = fo_loewner_factory(leftPoints,  rightPoints, ...
%                                                  leftWeights, rightWeights, ...
%                                                  leftSamples, rightSamples)
%
% DESCRIPTION:
%   This is a factory function to compute Loewner matrix quadruples
%   (LBAR, MBAR, HBAR, GBAR) for use in system realization algorithms: the
%   Loewner framework, and Quadrature-based Balanced Truncation.
%   See the paper "Data-Driven Balancing of Linear Dynamical Systems"
%   [Gosea, Gugercin, and Beattie 2022] for details.
%
%   To compute matrices for the second-order Loewner framework presents in
%   "A framework for the solution of the generalized realization problem" 
%   [Mayo and Antoulas 2007], set leftWeights and rightWeights to
%   be vectors of all ones.
%
% INPUTS:
%   leftPoints   - nLeft x 1 dimensional vector containing left 
%                  evaluation points
%   rightPoints  - nRight x 1 dimensional vector containing right 
%                  evaluation points
%   leftWeights  - nLeft x 1 dimensional vector containing left 
%                  weights for diagonal scaling
%   rightWeights - nRight x 1 dimensional vector containing right 
%                  weights for diagonal scaling
%   leftSamples  - p x m x nLeft dimensional array containing left 
%                  transfer function data sampled at leftPoints
%   rightSamples - p x m x nRight dimensional array containing right
%                  transfer function data sampled at rightPoints
%
% OUTPUTS: 
%   EBAR - nLeft*p x nRight*m diagonally scaled Loewner matrix; corresponds
%          to descriptor matrix E in system realization
%   ABAR - nLeft*p x nRight*m diagonally scaled shifted Loewner matrix;
%          corresponds to state matrix A in system realization 
%   BBAR - nLeft*p x m matrix of left data; corresponds to input matrix B 
%          in system realization
%   CBAR - p x nRight*m matrix of right data; corresponds to output matrix 
%          C in system realization
% 

%
% This file is part of the archive Code, Data, and Results for Numerical 
% Experiments in "..."
% Copyright (c) 2024 Sean Reiter, Steffen W. R. Werner
% All rights reserved.
% License: BSD 2-Clause license (see COPYING)
%
% Virginia Tech, USA
% Last editied: 9/4/2024
%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK INPUTS.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dimensions.
nLeft  = length(leftPoints);    
nRight = length(rightPoints);
[p, m] = size(leftSamples(:, :, 1));

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOEWNER FACTORY.                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Space allocation.
EBAR = zeros(nLeft*p, nRight*m);
ABAR = zeros(nLeft*p, nRight*m);
BBAR = zeros(nLeft*p, m);
CBAR = zeros(p,       nRight*m);

% Calculation of BBAR, CBAR.
for k = 1:nLeft
    BBAR((k - 1)*p + 1:k*p, :) = leftWeights(k)*leftSamples(:, :, k); 
end
for j = 1:nRight
    CBAR(:, (j - 1)*m + 1:j*m) = rightWeights(j)*rightSamples(:, :, j);  
end

% Block entrywise calculation of EBAR, ABAR.
for k = 1:nLeft
    for j = 1:nRight
        tmpDenom = leftPoints(k) - rightPoints(j);
        % For EBAR.
        EBAR((k - 1)*p + 1:k*p, (j - 1)*m + 1:j*m) = -leftWeights(k)*rightWeights(j) ...
            *(leftSamples(:, :, k) - rightSamples(:, :, j))./tmpDenom;

        % For ABAR.
        ABAR((k - 1)*p + 1:k*p, (j - 1)*m + 1:j*m) = -leftWeights(k)*rightWeights(j) ...
            *(leftPoints(k)*leftSamples(:, :, k) - rightPoints(j)*rightSamples(:, :, j))./tmpDenom;
    end
end
end