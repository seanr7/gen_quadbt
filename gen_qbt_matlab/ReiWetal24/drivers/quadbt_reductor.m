function [HSVS, Er, Ar, Br, Cr] = quadbt_reductor(LBAR, MBAR, HBAR, GBAR, r, ...
    tol)
%QUADBT_REDUCTOR Reductor function for quadrature-based balanced truncation
%                (QuadBT).
%
% SYNTAX:
%   [Er, Ar, Br, Cr] = QUADBT_REDUCTOR(LBAR, MBAR, HBAR, GBAR, r, tol)
%   [Er, Ar, Br, Cr] = QUADBT_REDUCTOR(LBAR, MBAR, HBAR, GBAR, r)
%
% DESCRIPTION:
%   This function implements the reductor portion of QuadBT, given a fixed
%   order of reduction r and Loewner matrices.
%
% INPUTS:
%   LBAR - Loewner matrix; used to realize Er and compute approximate
%          system Hankel singular values
%   MBAR - shifted Loewner matrix; used to realize Ar
%   HBAR - scaled matrix of left data; used to realize Br
%   GBAR - scaled matrix of right data; used to realize Cr
%   r    - order of reduction
%   tol  - truncation tolerance for singular values (default 1e-14)
%
% OUTPUTS: 
%   HSVS - approximate system Hankel singular values, computed from LBAR
%   Er   - r x r dimensional reduced descriptor matrix (enforced to be
%          identity)
%   Ar   - r x r dimensional reduced state matrix
%   Br   - r x m dimensional reduced input matrix
%   Cr   - p x r dimensional reduced output matrix
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

narginchk(5, 6);
if nargin < 6
    % Default tolerance.
    tol = 1e-14;
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REDUCTOR.                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Singular value decomposition.
[Z, S, Y] = svd(LBAR, 'econ');
HSVS      = diag(S);
cutOff    = find(HSVS>tol, 1,'last');
if cutOff < r
    fprintf(1, 'Desired order of reduction is too large! Trailing singular values fall below accepted tolerance.\n')
    r = cutOff;
    fprintf(1, 'Setting order of reduction to r = %d.\n', r)
end

% Projection.
Wt = S(1:r, 1:r)^(-1/2)*Z(:, 1:r)';
V  = Y(:, 1:r)*S(1:r, 1:r)^(-1/2);
Er = eye(r, r);
Ar = Wt*MBAR*V;
Br = Wt*HBAR;
Cr = GBAR*V; 