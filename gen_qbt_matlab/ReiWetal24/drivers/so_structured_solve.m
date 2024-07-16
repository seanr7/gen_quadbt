function [V] = so_structured_solve(Mso, Dso, Kso, Bso, Cso, s, structRHS)
%SECONDORDERSTRUCTUREDSOLVE Function to implement a linear solve for
% descriptor and mass matrices with second order system structure.
%
% SYNTAX:
%   [V] = SecondOrderStructuredSolve(Mso, Dso, Kso, Bso, Cso, s, structRHS)
%   [V] = SecondOrderStructuredSolve(Mso, Dso, Kso, Bso, Cso, s)
%
% DESCRIPTION:
%   Function to compute a single (2n x 2n) linear system solve of 
%   the form
%
%       V = (s*E - A)\R;    (0a)
%    or W = ((s*E - A)')\R; (0b)
%
%   where the mass matrix (A), descriptor matrix (E), and right 
%   hand side (R) are obtained from the first order realization of 
%   a second order system, and thus have the particular structure
%
%       E = [I  0;   0     Mso];        (1)
%       A = [I  0;  -Kso  -Dso];        (2)
%       R = [0; Bso]; or R = [Cso'; 0]; (3)
%
%   Via the Woodbury matrix identity and the inverse formula of a 
%   2 x 2 block matrix, V is instead computed in an equivalent way 
%   using only n x n linear solves.
%   Option 1: V = (s*E - A)\R for R = [0; Bso], then
%       
%       Z = (s*Mso + Dso)\Bso;                              (4a)
%       V = [(1/s)*(Z - ((s^2)*Mso + s*Dso + Kso)\(Kso*Z)); (4b)
%            s*((s^2)*Mso + s*Dso + Kso)\Bso];
% 
%   If W = ((s*E - A)')\R for R = [Cso'; 0], then
%
%       Z = ((conj(s)^2)*Mso + conj(s)*Dso + Kso)\(Kso*Cso'); (5a)
%       V = [(1/conj(s))*(Cso' - Kso*Z);                      (5b)
%            Z];               
%
%   It is assumed that the complex shift s is not a pole of the 
%   matrix pencil (s*E - A) and (s*M + D), and that s is strictly 
%   nonzero.
%
% INPUTS:
%   Mso       - sparse second order mass matrix with dimensions n x n in 
%               (1)
%   Dso       - sparse second order damping matrix with dimensions n x n 
%               in (2)
%   Kso       - sparse second order stiffness matrix with dimensions n x n 
%               in (2)
%   Bso       - sparse second order input matrix with dimensions n x m in 
%               (3)
%   Cso       - sparse second order position output matrix with dimensions 
%               p x n in (3)
%   s         - complex shift in linear solve
%   structRHS - boolean; do we solve system (0a) or (0b)? (default 0)
%                  0 if v = (s*E - A)\R with R = [0;   Bso];                            
%                  1 if w = ((s*E - A)')\R with R = [Cso'; 0];  
%
% OUTPUTS:
%   V - sparse solution to the linear system (0a) or (0b) with 
%       dimensions 2n x m or 2n x p computed accoding to (4a) and (4b) or 
%       (5a) and (5b)

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

if nargin < 6
    fprintf('Set default of structRHS = 0.\n')
    structRHS = 0;
end

% Dimensions.
[n, ~] = size(Mso);
[~, m] = size(Bso);
[p, ~] = size(Cso);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IMPLEMENT SOLVE.                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if structRHS == 0 % 
    % Option 1. V = (s*E - A)\R with B = [0;   Bso]
    Z  = (s*Mso + Dso)\(Bso); 
    V1 = (1/s).*(Z - ((s^2).*Mso + s.*Dso + Kso)\(Kso*Z));
    V2 = s.*(((s^2).*Mso + s.*Dso + Kso)\(Bso));
    V  = spalloc(2*n, m, nnz(V1) + nnz(V2));
else 
    % Option 2. W = ((s*E - A)')\R with R = [Cso'; 0]
    sconj = conj(s);    
    Z     = ((sconj^2).*Mso + sconj.*Dso + Kso)\(Cso');
    V1    = (1/sconj).*((Cso') - Kso*Z);
    V2    = Z;        
    V     = spalloc(2*n, p, nnz(V1) + nnz(V2));
end
% Return solution.
V(1:n, :)     = V1;
V(n+1:2*n, :) = V2;
end
