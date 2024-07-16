function [LBAR, MBAR, HBAR, GBAR] = loewner_factory(leftPoints,  rightPoints, ...
                                                    leftWeights, rightWeights, ...
                                                    leftData,    rightData)
%LOEWNER_FACTORY Factory function for Loewner matrix quadruples.
%
% SYNTAX:
%   [L, M, G, H] = loewner_factory(leftPoints,  rightPoints, ...
%                                  leftWeights, rightWeights, ...
%                                  leftData,    rightData)
%
% DESCRIPTION:
%   This is a factory function to compute Loewner matrix quadruples 
%   (LBAR, MBAR, HBAR, GBAR) with matrix-valued transfer function data. 
%   Matrices are computed according to:
%
%       LBAR((k - 1)*p + 1:k*p, (j - 1)*m + 1:j*m) = -leftWeights(k)*rightWeights(j) ...
%                   *((leftData(:, :, k) - rightData(:, :, j)) ...
%                   ./(leftPoints(k) - rightPoints(j));                  (1)
%       MBAR((k - 1)*p + 1:k*p, (j - 1)*m + 1:j*m) = -leftWeights(k)*rightWeights(j) ...
%                   *((leftPoints(k)*leftData(:, :, k) - rightPoints(j)*rightData(:, :, j)) ...
%                   ./(leftPoints(k) - rightPoints(j)));                 (2)
%       HBAR((k - 1)*p + 1:k*p, :) = leftWeights(k)*leftData(:, :, k);   (3)
%       GBAR(:, (j - 1)*m + 1:j*m) = rightWeights(j)*rightData(:, :, j); (4) 
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
%                  transfer function data
%   rightData    - p x m x nRight dimensional array containing right 
%                  transfer function data
%
% OUTPUTS: 
%   LBAR - nLeft*p x nRight*m diagonally scaled Loewner matrix in (1); 
%          corresponds to state matrix A in system realization 
%   MBAR - nLeft*p x nRight*m diagonally scaled shifted Loewner matrix in 
%          (2); corresponds to descriptor matrix E in system realization 
%   HBAR - nLeft*p x m matrix of left data; in (3) corresponds to input 
%          matrix B in system realization
%   GBAR - p x nRight*m matrix of right data in (4); corresponds to output 
%          matrix C in system realization
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

% Dimensions.
nLeft  = length(leftPoints);    
nRight = length(rightPoints);
[p, m] = size(leftData(:, :, 1));

% Assertions. ...

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOEWNER FACTORY.                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Space allocation.
LBAR = zeros(nLeft*p, nRight*m);
MBAR = zeros(nLeft*p, nRight*m);
HBAR = zeros(nLeft*p, m);
GBAR = zeros(p, nRight*m);

% Loewner quadruple construction.
for k = 1:nLeft
    for j = 1:nRight 
    % Block entrywise calculation of L and M. 
    denom = leftPoints(k) - rightPoints(j);
    LBAR((k - 1)*p + 1:k*p, (j - 1)*m + 1:j*m) = -leftWeights(k)*rightWeights(j) ...
        *(leftData(:, :, k) - rightData(:, :, j))/denom;
    MBAR((k - 1)*p + 1:k*p, (j - 1)*m + 1:j*m) = -leftWeights(k)*rightWeights(j) ...
        *(leftPoints(k)*leftData(:, :, k) - rightPoints(j)*rightData(:, :, j)) ...
        /denom;
    % At first pass over right points, compute G simultaneously.
    if k == 1
        % Block entrywise calculation of G. 
        GBAR(:, (j - 1)*m + 1:j*m) = rightWeights(j)*rightData(:, :, j);  
    end
    end
    % Block entrywise calculation of H. 
    HBAR((k - 1)*p + 1:k*p, :) = leftWeights(k)*leftData(:, :, k);  
end
end