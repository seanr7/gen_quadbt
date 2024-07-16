function [Z, Y, info] = freq_lyap_rksm(A, B, E, opts)
%FREQ_LYAP_RKSM Frequency-limited Lyapunov equation solver.
%
% SYNTAX:
%   [Z, Y, info] = FREQ_LYAP_RKSM(A, B, E)
%   [Z, Y, info] = FREQ_LYAP_RKSM(A, B, E, opts)
%
% DESCRIPTION:
%   This function uses a rational Krylov subspace method for solving the
%   Lyapunov-type equation of the frequency-limited controllability
%   Gramian
%
%       A*X*E' + E*X*A' + B*y' + y*B' = 0                               (1)
%
%   with y = real(1i/Pi * log((i*f1 El + Al) \(i*f2 El + Al))*Bl), for the
%   solution X = Z*Y*Z'.
%
% INPUT:
%   A    - matrix of dimension n x n from (1)
%   B    - matrix of dimension n x m from (1)
%   E    - symmetric positive definite matrix of dimension n x n from (1)
%   opts - struct, containing optional parameters for the algorithm:
%   +-----------------+---------------------------------------------------+
%   |    PARAMETER    |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | FctUpdate       | {0, 1}, if true the function evaluation f(A)B is  |
%   |                 | additionally computed in every step until the     |
%   |                 | complete Lyapunov equation is converged           |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | Freqs           | vector, frequency ranges to compute the equation  |
%   |                 | of with Freqs(k) < Freqs(k-1)                     |
%   |                 | (default [0, 1.0e+04])                            |
%   +-----------------+---------------------------------------------------+
%   | Info            | {0, 1, 2}, to disable/enable the normal or        |
%   |                 | extended info modes                               |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | MaxIter         | positive integer, maximum number of iteration     |
%   |                 | steps                                             |
%   |                 | (default min(floor(2*n/m), 100))                  |
%   +-----------------+---------------------------------------------------+
%   | MinIter         | positive integer, minimum number of iteration     |
%   |                 | steps                                             |
%   |                 | (default min(opts.MaxIter, 10))                   |
%   +-----------------+---------------------------------------------------+
%   | ModGramian      | {0, 1}, if true the modified version of the       |
%   |                 | matrix equation is computed (for stability        |
%   |                 | preservation)                                     |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | Npts            | positive integer, number of points used as        |
%   |                 | candidates for the adaptive shift computation     |
%   |                 | (default 1601)                                    |
%   +-----------------+---------------------------------------------------+
%   | S1              | real scalar, estimation of real part of smallest  |
%   |                 | eigenvalue                                        |
%   |                 | (default real(eigs(A,E,1,'SM')))                  |
%   +-----------------+---------------------------------------------------+
%   | S2              | real scalar, estimation of negative absolute value|
%   |                 | of largest eigenvalue                             |
%   |                 | (default -abs(eigs(A,E,1,'LM')))                  |
%   +-----------------+---------------------------------------------------+
%   | Shifts          | vector or character array, determining how the    |
%   |                 | shifts has to be chosen                           |
%   |                 |   [vector]    - precomputed shifts                |
%   |                 |   imaginary   - from the imaginary axis           |
%   |                 |   real        - from the real axis                |
%   |                 |   convex      - from convex hull of Ritz values   |
%   |                 |   convex_real - from real hull of Ritz values     |
%   |                 | (default 'imaginary')                             |
%   +-----------------+---------------------------------------------------+
%   | Solver          | character array, determining how the computation  |
%   |                 | of the matrix functional RHS f(A)B is done        |
%   |                 |   'logm'     - using the matrix logarithm         |
%   |                 |   'integral' - using the integral expression      |
%   |                 | (default 'logm')                                  |
%   +-----------------+---------------------------------------------------+
%   | SolverFreq      | positive integer, steps when to compute the       |
%   |                 | soultion of the projected matrix equation         |
%   |                 | (deault 1)                                        |
%   +-----------------+---------------------------------------------------+
%   | StoreFacE       | {0, 1}, disables/enables storing of the Cholesky  |
%   |                 | factors of the mass matrix                        |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | StoreV          | {0, 1}, turns storing the used projection basis   |
%   |                 | off and on                                        |
%   |                 | (default 0)                                       |
%   +-----------------+---------------------------------------------------+
%   | TolComp         | nonnegative scalar, tolerance for compression of  |
%   |                 | the solution factors                              |
%   |                 | (default eps)                                     |
%   +-----------------+---------------------------------------------------+
%   | TolLyap         | nonnegative scalar, tolerance for solution of     |
%   |                 | the matrix equation with approximated RHS         |
%   |                 | (default 1.0e-12)                                 |
%   +-----------------+---------------------------------------------------+
%   | TolRHS          | nonnegative scalar, tolerance for relative change |
%   |                 | of the RHS function evaluation f(A)b              |
%   |                 | (default 1.0e-08)                                 |
%   +-----------------+---------------------------------------------------+
%   | TrueRes         | {0, 1}, if true the residual corresponding to the |
%   |                 | mass matrix is computed instead of the            |
%   |                 | transformed one                                   |
%   |                 | (default 1)                                       |
%   +-----------------+---------------------------------------------------+
%   | L               | n x n matrix, left factor of the E matrix, such   |
%   |                 | that E(pL, pL) = L*L'                             |
%   |                 | (default [])                                      |
%   +-----------------+---------------------------------------------------+
%   | pL              | vector, row permutation vector of the first-order |
%   |                 | E matrix, such that E(pL, pL) = L*L'              |
%   |                 | (default [])                                      |
%   +-----------------+---------------------------------------------------+
%
% OUTPUT:
%   Z    - matrix of dimensions 2n x r, solutuion factor of X = Z*Y*Z'
%   Y    - matrix of dimensions r x r, solution factor of X = Z*Y*Z'
%   info - struct, containing the following information:
%   +-----------------+---------------------------------------------------+
%   |      ENTRY      |                     MEANING                       |
%   +-----------------+---------------------------------------------------+
%   | Errors          | matrix of dimensions 3 x k, containing the        |
%   |                 | indices of the iteration steps Errors(1, :),      |
%   |                 | the relative changes of the function evaluation   |
%   |                 | Errors(2, :) and the residuals of the matrix      |
%   |                 | equation Errors(3, :)                             |
%   +-----------------+---------------------------------------------------+
%   | FunAB           | matrix of dimensions n x m, the approximation of  |
%   |                 | the function evaluation f(A)B                     |
%   +-----------------+---------------------------------------------------+
%   | IterationSteps  | number of performed iteration steps               |
%   +-----------------+---------------------------------------------------+
%   | Shifts          | vector of used shifts                             |
%   +-----------------+---------------------------------------------------+
%   | V               | matrix of dimensions 2*n x l, the orthonormal     |
%   |                 | basis of the rational Krylov subspace, only if    |
%   |                 | StoreV == 1                                       |
%   +-----------------+---------------------------------------------------+
%   | L               | factor matrix of the E matrix such that           |
%   |                 | E(pL, pL) = L*L'                                  |
%   +-----------------+---------------------------------------------------+
%   | pL              | permutation vector of the E matrix such that      |
%   |                 | El(pL, pL) = L*L'                                 |
%   +-----------------+---------------------------------------------------+
%
%
% REFERENCE:
% P. Benner, P. Kuerschner, J. Saak, Frequency-limited balanced truncation
% with lowrank approximations, SIAM J. Sci. Comput. 38 (1) (2016)
% A471--A499 (Feb. 2016). doi:10.1137/15M1030911

%
% Copyright (C) 2019-2021 Peter Benner, Steffen W. R. Werner
%
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
% 1. Redistributions of source code must retain the above copyright notice,
%    this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in the
%    documentation and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECK INPUTS.                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

narginchk(3, 4);

if (nargin < 4) || isempty(opts)
    opts = struct();
end

% Check input matrices.
n = size(A, 1);

assert(isequal(size(A), [n n]), ...
    'The matrix A has to be square!');

assert(isequal(size(E), [n n]), ...
    'The matrix E must have the same dimensions as A!');

assert(norm(E - E', 'fro') < eps, ...
    'Only symmetric E are allowed!');

assert(size(B, 1) == n, ...
    'The matrix B must have the same number of rows as A!');

m = size(B, 2);

% Check and assign optional inputs.
if isfield(opts, 'FctUpdate') && not(isempty(opts.FctUpdate))
    assert((opts.FctUpdate == 0) || (opts.FctUpdate == 1), ...
        'The parameter ''FctUpdate'' has to be boolean!');
else
    opts.FctUpdate = 0;
end

if isfield(opts, 'Freqs') && not(isempty(opts.Freqs))
    assert(isvector(opts.Freqs) && (length(opts.Freqs) > 1), ...
        'The parameter ''Freqs'' has to be a vector!');
else
    opts.Freqs = [0, 1.0e+4];
end

if isfield(opts, 'Info') && not(isempty(opts.Info))
    assert((opts.Info == 0) || (opts.Info == 1) || (opts.Info == 2), ...
        'The parameter ''Info'' has to be 0, 1 or 2!')
else
    opts.Info = 0;
end

if isfield(opts, 'MaxIter') && not(isempty(opts.MaxIter))
    assert(opts.MaxIter > 0, ...
        'The parameter ''MaxIter'' must be a positive integer!');
else
    opts.MaxIter = min(floor(2*n/m), 100);
end

if isfield(opts, 'MinIter') && not(isempty(opts.MinIter))
    assert(opts.MinIter > 0, ...
        'The parameter ''MinIter'' must be a positive integer!');
    
    if opts.MinIter > opts.MaxIter
        warning(['Maximum number of iterations is too small.\n' ...
            ' Set to %d.'], opts.MinIter);
        opts.MaxIter = opts.MinIter;
    end
else
    opts.MinIter = min(opts.MaxIter, 10);
end

if isfield(opts, 'ModGramian') && not(isempty(opts.ModGramian))
    assert((opts.ModGramian == 0) || (opts.ModGramian == 1), ...
        'The parameter ''ModGramian'' has to be boolean!')
else
    opts.ModGramian = 0;
end

if isfield(opts, 'Npts') && not(isempty(opts.Npts))
    assert(not(mod(opts.Npts, 1)) && (opts.Npts > 0), ...
        'The parameter ''Npts'' has to be a positive integer!')
else
    opts.Npts = 1601;
end

if isfield(opts, 'Shifts') && not(isempty(opts.Shifts))
    assert(strcmpi(opts.Shifts, 'imaginary') ...
        || strcmpi(opts.Shifts, 'real') ...
        || strcmpi(opts.Shifts, 'convex') ...
        || strcmpi(opts.Shifts, 'convex_real') ...
        || isnumeric(opts.Shifts), ...
        'The requested shift version is not implemented!');
    
    if isnumeric(opts.Shifts)
        xi = opts.Shifts;
    else
        xi       = [];
        xi_cand  = [];
        nFreqs   = length(opts.Freqs) / 2;
        
        % Sample points.
        switch opts.Shifts
            case 'imaginary'
                for k = 1:nFreqs
                    xi_cand = [xi_cand, ...
                        1i * linspace(opts.Freqs(2 * k - 1), ...
                        opts.Freqs(2 * k), ...
                        ceil(opts.Npts / nFreqs))]; %#ok<AGROW>
                end
            case 'real'
                xi_cand  = logspace(-10, 10, opts.Npts);
        end
    end
else
    opts.Shifts = 'imaginary';
    xi          = [];
    xi_cand     = [];
    nFreqs      = length(opts.Freqs) / 2;
    
    for k = 1:nFreqs
        xi_cand = [xi_cand, ...
            1i * linspace(opts.Freqs(2 * k - 1), ...
            opts.Freqs(2 * k), ...
            ceil(opts.Npts / nFreqs))]; %#ok<AGROW>
    end
end

if isfield(opts, 'Solver') && not(isempty(opts.Solver))
    assert(strcmpi(opts.Solver, 'logm') ...
        || strcmpi(opts.Solver, 'integral'), ...
        'The requested solver is not implemented!')
else
    opts.Solver = 'logm';
end

if isfield(opts, 'SolverFreq') && not(isempty(opts.SolverFreq))
    assert(opts.SolverFreq > 0, ...
        'The parameter ''SolverFreq'' must be a positive integer!');
else
    opts.SolverFreq = 1;
end

if isfield(opts, 'StoreFacE') && not(isempty(opts.StoreFacE))
    assert((opts.StoreFacE == 0) || (opts.StoreFacE == 1), ...
        'The parameter ''StoreFacE'' has to be boolean!')
else
    opts.StoreFacE = 0;
end

if isfield(opts, 'StoreV') && not(isempty(opts.StoreV))
    assert((opts.StoreV == 0) || (opts.StoreV == 1), ...
        'The parameter ''StoreV'' has to be boolean!')
else
    opts.StoreV = 0;
end

if isfield(opts, 'TolComp') && not(isempty(opts.TolComp))
    assert(opts.TolComp >= 0, ...
        'The parameter ''TolComp'' has to be larger or equal to zero!');
else
    opts.TolComp = eps;
end

if isfield(opts, 'TolLyap') && not(isempty(opts.TolLyap))
    assert(opts.TolLyap >= 0, ...
        'The parameter ''TolLyap'' has to be larger or equal to zero!');
else
    opts.TolLyap = 1.0e-12;
end

if isfield(opts, 'TolRHS') && not(isempty(opts.TolRHS))
    assert(opts.TolRHS >= 0, ...
        'The parameter ''TolRHS'' has to be larger or equal to zero!');
else
    opts.TolRHS = 1.0e-08;
end

if isfield(opts, 'TrueRes') && not(isempty(opts.TrueRes))
    assert((opts.TrueRes == 0) || (opts.TrueRes == 1), ...
        'The parameter ''TrueRes'' has to be boolean!');
else
    opts.TrueRes = 1;
end

% Check for given factorizations of the mass matrix.
if isfield(opts, 'L') && not(isempty(opts.L))
    assert(isequal(size(opts.L), [n n]), ...
        'The factor L must have the same dimensions as E!');
    L = opts.L;
else
    L = [];
end

if isfield(opts, 'pL') && not(isempty(opts.pL))
    assert(isvector(opts.pL), ...
        'The parameter ''pL'' has to be a permutation vector!');
    
    pL = opts.pL;
else
    pL = [];
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if opts.Info == 2
    fprintf(1, 'Initialization\n');
    fprintf(1, '==============\n');
end

% Factorization of the E matrix.
if isempty(L)
    spSolve = issparse(E);
else
    spSolve = issparse(L);
end

if isempty(L)
    if opts.Info == 2
        fprintf(1, '\t Compute Cholesky factorization.\n');
    end
    
    if spSolve
        [L, q, pL] = chol(E, 'lower', 'vector');
        
        if opts.Info == 2
            fprintf(1, '\t Cholesky factor NNZ: %d.\n', nnz(L));
        end
    else
        [L, q] = chol(E, 'lower');
    end
    
    assert(q == 0, 'The E matrix is not invertible!');
end

% Scaling factor for matrix functional.
if (opts.Freqs(1) == 0) || ((length(opts.Freqs) > 1) ...
        && (opts.Freqs(1) == -opts.Freqs(2)))
    opts.Freqs(1)    = -opts.Freqs(2);
    scaling          = 1 / (2 * pi);
else
    scaling = 1 / pi;
end

% Flipping matrix.
flip = [zeros(m), eye(m); eye(m), zeros(m)];
resM = [zeros(m), eye(m); eye(m), zeros(m)];

% Variable for modified Gramian computation.
if opts.ModGramian
    mod_gramian_rhs = 0;
    if opts.FctUpdate
        warning('FREQ_LYAP_RKSM_SO:ModGramian', ...
            ['Update of Function evaluation is not implemented ' ...
            'for the modified Gramian. opts.FctUpdate set to 0.']);
        opts.FctUpdate = 0;
    end
end

% Convergence variables.
conv_fAb  = 0;
conv_lyap = 0;

% Storage for the errors.
err = zeros(3, opts.MaxIter);

% Storage for updating vectors.
w     = zeros(n, 2 * m);
tmp   = zeros(n, 2 * m);
funAb = zeros(n, m);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STARTING PHASE.                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Transformed right hand-side.
if spSolve
    vE = L \ B(pL, :);
else
    vE = L \ B;
end

% Orthogonalization of right hand-side.
[vE, beta] = qr(vE, 0);

% Estimation of first shifts.
orig_state = warning('off', 'MATLAB:normest:notconverge');
eigopts = struct('cholB', 1, 'tol', 1.0e-03, 'maxit', 10);
if spSolve
    eigopts.permB = pL;
end

if opts.Info == 2
    fprintf(1, '\t Computation of initial adaptive Shifts.\n');
    eigopts.disp = 2;
end

if isfield(opts, 'S1') && not(isempty(opts.S1))
    assert(isscalar(opts.S1) && isreal(opts.S1), ...
        'Parameter S1 has to be a real scalar!');
    
    s1 = opts.S1;
else
    try
        s1 = real(eigs(A, L', 1, 'SM', eigopts));
    catch
        s1 = -0.01;
    end
    
    if isempty(s1) || isnan(s1)
        s1 = -0.01;
    end
end

if isfield(opts, 'S2') && not(isempty(opts.S2))
    assert(isscalar(opts.S2) && isreal(opts.S2), ...
        'Parameter S2 has to be a real scalar!');
    
    s2 = opts.S2;
else
    try
        s2 = -abs(eigs(A, L', 1, 'LM', eigopts));
    catch
        s2 = -normest(A);
    end
        
    if isempty(s2) || isnan(s2)
        s2 = -normest(A);
    end
end
warning(orig_state);

% Set up projection matrices.
if opts.Info == 2
    fprintf(1, '\t Allocate memory for basis.\n');
end
V = zeros(n, m * 2 * opts.MaxIter + 1);

if opts.Info == 2
    fprintf(1, '\t Allocate memory for Rayleigh quotient.\n');
end
AV = V;

V(:, 1:m) = vE;

% Initial Rayleigh quotient.
if spSolve
    tmp(pL, 1:m) = L' \ vE;
    tmp(:, 1:m)  = A * tmp(:, 1:m);
    AV(:, 1:m)   = L \ tmp(pL, 1:m);
else
    AV(:, 1:m) = L \ (A * (L' \ vE));
end

Rt    = vE' * AV(:, 1:m);                            % initial projected A = V' * (L \ A / L') * V
theta = eig(Rt);                                     % initial Ritz values
H     = zeros(2*m*opts.MaxIter+1, 2*m*opts.MaxIter); % initial Hessenberg matrix

fmR0  = [];          % previous f(A)B term
fmR   = zeros(n, 1); % initial f(A)B term
xi    = xi(:).';     % reshaping of shift vector

niter      = 1;  % overall iteration counter
jc         = 1;  % inner iteration counter for storing the errors
curr_start = 1;  % current index in projection matrices
sigma      = -1; % selected shift


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ARNOLDI ITEARTION.                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if opts.Info == 2
    fprintf(1, '\n');
    fprintf(1, 'Arnoldi Iteration\n');
    fprintf(1, '=================\n');
end

while niter <= opts.MaxIter
    
    % Indices for all the dimensions of the projection matrices.
    prev_start = curr_start;         % .. The first index where columns were added in the previous step.
    prev_end   = prev_start + m - 1; % .. The last index where columns were added in the previous step. [curr_start, curr_end]
    real_start = prev_end + 1;       % .. The first index where columns (for the real part) will be added in the current step.
    real_end   = real_start + m - 1; % .. The last index where columns (for the real part) will be added in the current step. [real_start, real_end].
    imag_start = real_end + 1;       % .. The first index where columns for the imaginary part will be added in the current step if the shift is complex.
    imag_end   = imag_start + m - 1; % .. The last index where columns for the imaginary part will be added in the current step if the shift is complex. [imag_start, imag_end].
    
    js         = niter * m; % Additional counter for size test.
    sigma_prev = sigma;     % Previous shift.
    
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SHIFT COMPUTATION.                                                  %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if opts.Info == 2
        fprintf(1, '\t Compute next shift: ');
    end
    
    if not(isnumeric(opts.Shifts))
        % Mirror anti-stable Ritz values.
        theta(real(theta) > 0) = -theta(real(theta) > 0);
        
        % Make shifts real for real convex hull.
        if strcmpi(opts.Shifts, 'convex_real')
            theta = real(theta);
        end
        
        thetao = theta;
        if strcmpi(opts.Shifts, 'convex') ...
                || strcmpi(opts.Shifts, 'convex_real')
            %  Complex convex hull test set ala  Simoncini/Druskin.
            if any(imag(theta)) && (length(theta) > 2)
                theta(end+1) = s2; %#ok<AGROW>                     % Extend computed shifts by negative norm estimation.
                ch           = convhull(real(theta), imag(theta)); % Convex hull of shifts
                eH           = -theta(ch);                         % Mirrored Ritz values at boundary.
                ieH          = length(eH);                         % Number of remaining shifts.
                missing      = niter * m - ieH;                    % Number of lost shifts.
                
                % Include enough points from the border (for MIMO).
                while missing > 0
                    neweH   = (eH(1:ieH-1) + eH(2:ieH)) / 2;
                    eH      = [eH; neweH]; %#ok<AGROW>
                    missing = niter * m - length(eH);
                end
            else
                % Convex hull of real shifts by artificial extension.
                eH = sort([-real(theta); -s1; -s2]);
            end
            
            if niter == 1
                % Simply take the bounds in the first iteration.
                eH = sort(-[s1; s2]);
            end
            
            xi_cand = [];
            for j = 1:length(eH)-1
                % Candidate set by points from convex hull.
                xi_cand = [xi_cand, linspace(eH(j), eH(j+1), 500 / m)]; %#ok<AGROW>
            end
        end
        
        if niter == 1
            % Only in the first iteration step.
            if strcmpi(opts.Shifts, 'convex') ...
                    || strcmpi(opts.Shifts, 'convex_real')
                gs = -s1 * ones(m, 1);
            else
                gs = inf * ones(m, 1);
            end
        else
            % In all further iteration steps.
            gs = kron(xi(2:end), ones(1, m))';
        end
        
        % Evaluate the rational function for all the candidate values for
        % the shifts and find afterwards the maximum. (Time-Limited (16))
        sn      = ratfun(xi_cand, thetao, gs);
        [~, jx] = max(abs(sn)); % Get index of the maximal function value.
        if real(xi_cand(jx)) < 0
            % Mirror unstable optimal shift.
            xi_cand(jx) = -xi_cand(jx); %#ok<AGROW>
        end
        
        % Remove imaginary dirt and set next shift.
        if abs(imag(xi_cand(jx))) / abs(xi_cand(jx)) < 1.0e-08
            xi(niter + 1) = real(xi_cand(jx));
        else
            xi(niter + 1) = xi_cand(jx);
        end
    end
    
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SOLVE SHIFTED LINEAR SYSTEM.                                        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if isnumeric(opts.Shifts)
        sigma = xi(niter);
    else
        sigma = xi(niter + 1);
    end
    
    if opts.Info == 2
        fprintf(1, '%e + %ei\n', real(sigma), imag(sigma));
    end
    
    % If only conjugate of the previous shift, take the next one.
    if sigma_prev == conj(sigma)
        continue;
    end
    
    % Solve linear system.
    if opts.Info == 2
        fprintf(1, '\t Solve shifted linear system.\n');
    end
    
    k = length(prev_start:prev_end);
    if abs(sigma) > eps
        if spSolve
            tmp(pL, 1:k) = L * V(:, prev_start:prev_end);
            tmp(:, 1:k)  = (A - sigma * E) \ tmp(:, 1:k);
            w(:, 1:k)    = L' * tmp(pL, 1:k);
        else
            w(:, 1:k) = L' * ((A - sigma * E) ...
                \ (L * V(:, prev_start:prev_end)));
        end
    else
        sigma = 0;
        
        if spSolve
            tmp(pL, 1:k) = L * V(:, prev_start:prev_end);
            tmp(:, 1:k)  = A \ tmp(:, 1:k);
            w(:, 1:k)    = L' * tmp(pL, 1:k);
        else
            w(:, 1:k) = L' * (A \ (L * V(:, prev_start:prev_end)));
        end
    end
    
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BASIS EXPANSION.                                                    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if opts.Info == 2
        fprintf(1, '\t Basis expansion by orthogonalization.\n');
    end
    
    % Expansion by real part.
    wR = real(w(:, 1:k));
    for it = 1:2 % Repeated modified block Gram-Schmidt.
        for kk = 1:niter
            k1    = (kk - 1) * m + 1;
            k2    = kk * m;
            gamma = V(:, k1:k2)' * wR;
            
            H(k1:k2, real_start-m:real_end-m) = ...
                H(k1:k2, real_start-m:real_end-m) + gamma;
            wR = wR - V(:, k1:k2) * gamma;
        end
    end
    
    [V(:, real_start:real_end), ...
        H(real_start:real_end, real_start-m:real_end-m)] ...
        = qr(wR, 0);
    
    % Expansion by imaginary part.
    if not(isreal(sigma))
        wR = imag(w(:, 1:k));
        
        for it = 1:2 % Repeated modified block Gram-Schmidt.
            for kk = 1:niter+1
                k1    = (kk - 1) * m + 1;
                k2    = kk * m;
                gamma = V(:, k1:k2)' * wR;
                
                H(k1:k2, imag_start-m:imag_end-m) = ...
                    H(k1:k2, imag_start-m:imag_end-m) + gamma;
                wR = wR - V(:, k1:k2) * gamma;
            end
        end
        
        [V(:,imag_start:imag_end), ...
            H(imag_start:imag_end, imag_start-m:imag_end-m)] ...
            = qr(wR, 0);
        
        % Update the indices by real and imaginary expansion.
        curr_start = real_start;
        curr_end   = imag_end;
        proj_end   = real_end;
    else
        % Update the indices by real expansion.
        curr_start = real_start;
        curr_end   = real_end;
        proj_end   = prev_end;
    end
    
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATE RAYLEIGH QUOTIENT.                                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if opts.Info == 2
        fprintf(1, '\t Update Rayleigh quotient.\n');
    end
    
    k = length(curr_start:curr_end);
    if spSolve
        tmp(pL, 1:k) = L' \ V(:, curr_start:curr_end);
        tmp(:, 1:k)  = A * tmp(:, 1:k);
        
        AV(:, curr_start:curr_end) = L \ tmp(pL, 1:k);
    else
        AV(:, curr_start:curr_end) = L \ (A * (L' ...
            \ V(:, curr_start:curr_end)));
    end
    
    % Extension of the Rayliegh quotient matrix.
    g  = V(:, 1:prev_end)' * AV(:, curr_start:curr_end);
    g3 = V(:, curr_start:curr_end)' * AV(:, 1:curr_end);
    Rt = [Rt, g; g3]; %#ok<AGROW>
    
    % Set indices and the second complex shift.
    if not(isreal(sigma))
        % Two steps were done at once.
        niter      = niter + 1;
        curr_start = imag_start;
        if not(isnumeric(opts.Shifts))
            % Set the second shift as complex conjugate.
            xi(niter + 1) = conj(xi(niter));
        end
    end
    
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SOLVING PROJECTED PROBLEM.                                          %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if opts.Info == 2
        fprintf(1, '\t Solve projected problem.\n');
    end
    
    cplxssolve = ((niter > 1) && (xi(niter - 1) == conj(xi(niter))) ...
        && mod(niter - 1, opts.SolverFreq) == 0);
    
    if cplxssolve || (mod(niter, opts.SolverFreq) == 0)
        err(1, jc) = niter;% Indices of steps.
        
        % Check if f(A)B already converged.
        if not(conv_fAb) || opts.FctUpdate
            % Compute small right hand-side.
            fmR = projected_freq( ...
                Rt(1:proj_end, 1:proj_end), ...
                [beta; zeros(proj_end-m, m)], ...
                opts.Freqs, length(opts.Freqs) / 2, scaling, opts.Solver);
            df  = size(fmR, 1) - size(fmR0, 1);
            
            % Relative change of matrix functional.
            if norm(fmR, 2) == 0
                err(2, jc) = norm([fmR0; zeros(df, m)] - fmR, 2);
            else
                err(2, jc) = norm([fmR0; zeros(df, m)] - fmR, 2) ...
                    / norm(fmR, 2);
            end
            
            % Print information.
            if opts.Info > 0
                fprintf('Step %3d -- %1.6e       %d\n', ...
                    niter, err(2, jc), niter * m);
            end
            
            % Back projection if right hand-side approximation good enough.
            if (err(2, jc) < opts.TolRHS) || (niter >= opts.MaxIter)
                funAb_ss = V(:, 1:proj_end) * fmR;
                
                if spSolve
                    funAb(pL, :) = L * funAb_ss;
                else
                    funAb = L * funAb_ss;
                end
                conv_fAb = 1;
                
            end
            fmR0            = fmR;
            mod_gramian_rhs = 0;
        end
        
        % Solve complete Lyapunov equation with f(A)B.
        if ((cplxssolve || (mod(niter, opts.SolverFreq) == 0)) ...
                && conv_fAb) || (niter >= opts.MaxIter)
            if opts.TolLyap
                % Switch to adaptive convex hull shifts.
                if isnumeric(opts.Shifts) ...
                        || strcmpi(opts.Shifts, 'imaginary')
                    
                    opts.Shifts = 'convex';
                    xi          = [1, xi(1:niter)];
                end
                
                % Back-projected f(A)B if necessary.
                if js == size(fmR, 1)
                    yt = fmR;
                else
                    yt = V(:, 1:proj_end)' * funAb_ss;
                end
                
                % Modified frequency-limited Gramians.
                if opts.ModGramian
                    eigsops.issym  = 1;
                    eigsops.tol    = 1e-10;
                    eigsops.isreal = 1;
                    
                    % Build the RHS factor of the modified fl CALEs.
                    if not(mod_gramian_rhs)
                        % LDL_T Compression of right hand-side.
                        [Db, Eb] = eigs(@(x)[B, funAb] * (flip ...
                            * ([B, funAb]' * x)), n, 2 * m, ...
                            'LM', eigsops);
                        e        = diag(Eb);
                        [e, idx] = sort(abs(e),'descend');
                        Db       = Db(:,idx);
                        beta     = (Db(:,1:2*m) * diag(sqrt(e(1:2*m))));
                        
                        if spSolve
                            Ba = L \ beta(pL, :);
                        else
                            Ba = L \ beta;
                        end
                        
                        mod_gramian_rhs = 1;
                        funAb           = beta;
                        
                        if not(isnumeric(opts.Shifts))
                            opts.Shifts = 'convex';
                        end
                    end
                    
                    Ba_p = V(:, 1:proj_end)' * Ba;
                    rhs2 = Ba_p * Ba_p';
                    nBf  = norm(rhs2, 2);
                else
                    roff = beta * yt(m+1:end, :)';
                    rhs2 = [yt(1:m, 1:m) * beta' + beta * yt(1:m, 1:m)',...
                        roff; roff', zeros(proj_end-m, proj_end-m)];
                    nBf  = norm(rhs2, 2); % ORIGINAL CODE 
                    % Try and catch implemented due to error (SR,
                    % 6-26-2024)
                    % try
                    %     nBf  = norm(rhs2, 2); % ORIGINAL CODE 
                    % catch
                    %     nBf  = normest(rhs2, 2);
                    % end
                end
                
                % Solve the projected Lyapunov equation.
                Y = lyap(Rt(1:proj_end, 1:proj_end), rhs2);
                
                % Computed residual.
                g1 = V(:, 1:curr_end-m)' ...
                    * AV(:, curr_end-m+1:curr_end);
                u1 = AV(:, curr_end-m+1:curr_end) ...
                    - V(:, 1:curr_end-m) * g1;
                HT = H(curr_end-m+1:curr_end, proj_end-m+1:proj_end);
                
                if opts.TrueRes
                    % Residual of original equation.
                    if isreal(sigma)
                        % Real case.
                        uf      = V(:, curr_end-m+1:curr_end) * sigma - u1;
                        hw      = V(:,1:proj_end) ...
                            * (([zeros(m, proj_end-m), HT] ...
                            / H(1:proj_end, 1:proj_end))  *Y)';
                        [~, g3] = qr(L * [uf, hw], 0);
                        
                        err(3, jc) = norm(g3 * (resM * g3'), 2) / nBf;
                    else
                        % Complex case.
                        y1 = [zeros(m, proj_end-m), HT];
                        y2 = [zeros(m, proj_end-2*m), ...
                            -imag(sigma) * HT, real(sigma) * HT];
                        hw = V(:,1:proj_end) * (([y2; -y1] ...
                            / H(1:proj_end, 1:proj_end)) * Y)';
                        
                        [~, g3] = qr(L ...
                            * [V(:, curr_end-m+1:curr_end), u1, hw], 0);
                        
                        err(3, jc) = norm(g3 * (kron(resM, eye(2)) ...
                            * g3'), 2) / nBf;
                    end
                else
                    % Residual of transformed equation.
                    if isreal(sigma)
                        % Real case.
                        [~, uu] = qr(V(:, curr_end-m+1:curr_end) ...
                            * sigma - u1, 0);
                        qf      = (uu * ([zeros(m, proj_end-m) HT] ...
                            / H(1:proj_end, 1:proj_end))) * Y;
                    else
                        % Complex case.
                        y1      = [zeros(m, proj_end-m), HT];
                        y2      = [zeros(m, proj_end-2*m), ...
                            -imag(sigma)*HT, real(sigma)*HT];
                        [~, uu] = qr([V(:, curr_end-m+1:curr_end), u1], 0);
                        qf      = uu * ([y2; -y1] ...
                            / H(1:proj_end, 1:proj_end)) * Y;
                    end
                    
                    err(3, jc) = norm(qf, 2) / nBf;
                end
                
                % Print residual info.
                if opts.Info > 0
                    fprintf('Lyap:  %3d\t %e\t %d\n', ...
                        niter, err(3, jc), niter * m);
                end
                
                % If residual is small enough.
                if (niter >= opts.MinIter) && (err(3, jc) < opts.TolLyap)
                    if opts.TolComp % Column compression of solution ZIZ'.
                        [uY, sY, ~] = svd(Y);
                        sY          = diag(sY);
                        is          = find(sY > opts.TolComp * sY(1));
                        Y           = uY(:, is) * diag(sqrt(sY(is)));
                        Z           = V(:, 1:size(Y, 1)) * Y;
                        
                        if spSolve
                            Z(pL, :) = L' \ Z;
                        else
                            Z = L' \ Z;
                        end
                        
                        Y = eye(size(Z, 2));
                    else % Non-compressed solution ZYZ'.
                        if spSolve
                            Z        = zeros(size(V(:, 1:size(Y, 1))));
                            Z(pL, :) = L' \ V(:, 1:size(Y, 1));
                        else
                            Z = L' \ V(:, 1:size(Y, 1));
                        end
                    end
                    V         = V(:, 1:proj_end);
                    conv_lyap = 1;
                    break;
                end
            else
                % Only interested in projection matrix.
                Z = [];
                Y = [];
                V = V(:, 1:proj_end);
                break;
            end
        end
        jc = jc + 1;
    end
    
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % UPDATE FOR NEXT ITERATION.                                          %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    theta = eig(Rt(1:proj_end, 1:proj_end));
    niter = niter + 1;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HANDLING OF NON-CONVERGED ITERATION.                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if niter > opts.MaxIter
    if not(conv_fAb)
        % Case: No converged f(A)B evaluation.
        % Evalute the last projected frequency-limited term.
        fmR = V(:,1:proj_end) * projected_freq( ...
            Rt(1:proj_end, 1:proj_end), ...
            [beta;zeros(proj_end-m,m)], ...
            opts.Freqs, length(opts.Freqs) / 2, scaling, opts.Solver);
        
        if spSolve
            funAb(pL, :) = L * fmR;
        else
            funAb = L * fmR;
        end
        
        warning(['No convergence of f(A)B evaluation in ' ...
            '%d iteration steps!'], ...
            niter - 1);
    elseif not(conv_lyap)
        % Case: Convergence in f(A)B evaluation, get current solution.
        if spSolve
            Z        = zeros(size(V(:, 1:size(Y, 1))));
            Z(pL, :) = L' \ V(:, 1:size(Y, 1));
        else
            Z = L' \ V(:, 1:size(Y, 1));
        end
        
        warning(['No convergence of Lyapunov equation solver in ' ...
            '%d iteration steps!'], ...
            niter - 1);
    end
    
    V = V(:,1:proj_end);
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADDITIONAL OUTPUT INFORMATION.                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

info = struct( ...
    'Errors'        , err(1:3, 1:find(err(1,:), 1, 'last')), ...
    'FunAB'         , funAb, ...
    'IterationSteps', niter, ...
    'Shifts'        , xi);

if opts.StoreFacE
    info.L  = L;
    info.pL = pL;
end

if opts.StoreV
    info.V = V;
end

end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTION: RATFUN                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = ratfun(x, eH, s)
%RATFUN Computes the values of a rational polynomial.
%
% DESCRIPTION:
%   This function computes for all given points the rational polynomial
%
%       f(x_j) = | prod_i=1^k (x_j - s_i)/(x_j - eHi) |
%
%   for all x_j, j = 1, ..., n.
%
% INPUT:
%   x  - vector of points, where to evaluate the polynomial
%   eH - vector of coefficients from the denominator
%   s  - vector of coefficients from the nominator
%
% OUTPUT:
%   r  - vector of function values.
%
%
% Steffen W. R. Werner, 2018.

r = zeros(1, length(x));
for j = 1:length(x)
    r(j) = abs(prod((x(j) - s) ./ (x(j) - eH)));
end

end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTION: PROJECTED_FREQ                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function F = projected_freq(R, projrhs, f, nF, scaling, ssolve)
%PROJECTED_FREQ Computes the projected frequency-limited right hand-side.
%
% DESCRIPTION:
%   This function solves the matrix functional f(A)B for small problems
%   using either the integral expression or the principal matrix logarithm.
%
% INPUT:
%   R       - projected A matrix
%   projrhs - projected B matrix
%   f       - frequency-range [f(1), f(2), ...]
%   nF      - number of Frequency-ranges
%   scaling - scaling parameter 1/pi or 1/(2*pi)
%   ssolve  - method for solving
%               'logm'     - principal matrix logarithm
%               'integral' - integral method
%
% OUTPUT:
%   F       - projected right hand-side of frequency-limited equation

js = size(R, 1);
jb = size(projrhs, 2);

% Solve via principal matrix logarithm.
if strcmp(ssolve, 'logm')
    C = eye(js);
    for u = 1:nF
        % Accumulate right hand-side for all frequency-ranges.
        C = C * ((R + 1i*f(2 * u - 1) * eye(js)) ...
            \ (R + 1i*f(2 * u) * eye(js)));
    end
    F = real(1i*scaling * logm(C) * projrhs);
end

% Solve via integral expression.
if strcmp(ssolve, 'integral')
    F = zeros(js, jb);
    for u = 1:nF
        % Accumulate right hand-side for all frequency-ranges.
        F = F + real(scaling*integral(@(t)(1i*t*eye(js) - R) \ projrhs, ...
            f(2 * u - 1), f(2 * u), 'ArrayValued', true, 'RelTol', 1e-2));
    end
end

end
