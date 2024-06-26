classdef flQuadBTSampler_so < handle
% FLQUADBTSAMPLER_so Specialized implementation of flQuadBTSampler class
% for first-order descriptor systems with underlying second-order
% structure. 
%
% DESCRIPTION:
%   Class is outfitted to compute transfer function samples for use in
%   frequency-limited Quadrature-based Balanced Truncation (flQuadBT) when
%   the underlying full-order model has sparse second-order structure.
%
% PROPERTIES:
%   Mso  - sparse second order mass matrix (n x n)
%   Dso  - sparse second order damping matrix (n x n)
%   Kso  - sparse second order stiffness matrix (n x n)
%   Bso  - sparse second order input matrix (n x m)  
%   Cso  - sparse second-order position output matrix (p x n)
%   n    - second-order state dimension
%   m    - second-order input dimension
%   p    - second-order output dimension
%   typ  - type of balancing beind done; `flbt'
%   time - boolean; do we time linear solves? (for truly large-scale
%          problems)
%

%
% This file is part of the archive Code, Data, and Results for Numerical 
% Experiments in "..."
% Copyright (c) 2024 Sean Reiter, Steffen W. R. Werner
% All rights reserved.
% License: BSD 2-Clause license (see COPYING)
%

% Virginia Tech, Department of Mathematics
% Last editied: 6/25/2024

    % See class description for details of properties.
    properties 
        Mso 
        Dso 
        Kso 
        Bso 
        Cso 
        n
        m 
        p 
        typ
        time
    end
    
    methods

    % Constructor method
    function obj = flQuadBTSampler_so(Mso, Dso, Kso, Bso, Cso, n, m, p, time)
        obj.Mso = Mso;  obj.Dso = Dso;  obj.Kso = Kso; 
        obj.Bso = Bso;  obj.Cso = Cso;

        % System matrices are sparse so state, input, output dimensions
        % are passed as arguments.
        obj.n = n;      obj.m = m;      obj.p = p;

        % Specify type of balancing so sampler class can play with reductor
        % class
        obj.typ  = 'flbt';
        obj.time = time;
    end

    %%
    %                              
    %   | | _|_ o | o _|_ o  _   _ 
    %   |_|  |_ | | |  |_ | (/_ _> 
    %          
    %%

    function [V] = so_structured_solve(obj, s, struct_rhs, time)
        % SO_STRUCTURED_SOLVE Function to implement a linear solve for
        % descriptor and mass matrices with second order system structure.
        %
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
        %       E = [I  0;   0     Mso];       (1)
        %       A = [I  0;  -Kso  -Dso];       (2)
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
        %   obj       - instance of sampler class containing second-order
        %               system matrices and relevant dimensions
        %   s         - complex shift in linear solve
        %   solve_opt - boolean, do we solve system (0a) or (0b)?
        %                  0 if v = (s*E - A)\R with R = [0;   Bso];                            
        %                  1 if w = ((s*E - A)')\R with R = [Cso'; 0];  
        %   time       - optional boolean argument to print time required
        %
        % OUTPUTS:
        %   V - sparse solution to the linear system (0a) or (0b) with 
        %       dimensions 2n x 1 computed accoding to (4a) and (4b) or 
        %       (5a) and (5b)
        
        % Last editied: 6/25/2024
    
        %
        % Check and set inputs
        if nargin < 4
            % Default is to not time solutions
            time = 0;
        end
        if nargin < 3
            error('Must specify structure of the right hand side!\n')
        end
        
        %
        if time
            tic
            fprintf(1, 'Initiating structured linear solve\n')
            fprintf(1, '----------------------------------\n')
        end
        
        % Structured solve; option 1 in (4a), (4b)
        if struct_rhs == 0 % if V = (s*E - A)\R with B = [0;   Bso]
            Z  = (s*obj.Mso + obj.Dso)\(obj.Bso); 
            V1 = (1/s).*(Z - ((s^2).*obj.Mso + s.*obj.Dso + obj.Kso)\(obj.Kso*Z));
            V2 = s.*(((s^2).*obj.Mso + s.*obj.Dso + obj.Kso)\(obj.Bso));
            V  = spalloc(2*obj.n, obj.m, nnz(V1) + nnz(V2));
        else % if W = ((s*E - A)')\R with R = [Cso'; 0]
            sconj = conj(s);    
            Z     = ((sconj^2).*obj.Mso + sconj.*obj.Dso + obj.Kso)\(obj.Cso');
            V1    = (1/sconj).*((obj.Cso') - obj.Kso*Z);
            V2    = Z;        
            V     = spalloc(2*obj.n, obj.p, nnz(V1) + nnz(V2));
        end
        V(1:obj.n, :)         = V1;
        V(obj.n+1:2*obj.n, :) = V2;
        
        if time
            fprintf(1, 'Structured solve finished in %.2f s\n',toc)
            fprintf(1, '-----------------------------------------\n');
        end

    end

    %%

    function Uquad = quad_right_factor(obj, nodesr, weightsr)
        % QUAD_RIGHT_FACTOR Function to explicitly compute the
        % quadrature-based approximation to the frequency-limited
        % Gramian P ~ Uquad*Uquad'
        %
        %
        % DESCRIPTION:
        %   Function computes multiple linear solves using the class 
        %   method `so_structure_solve' to compute the quadrature-based
        %   square root factor Uquad (n x m*J), J = length(nodesr).
        %   The jth (n x m) block of Uquad is defined by:
        %
        %       V = weightsr(j)*(nodesr(j)*Efo - Afo)\Bfo
        %
        %   Efo, Afo, and Bfo are the descriptor, mass, and input
        %   matrices obtained from the first-order realization of the
        %   underlying model. 
        %
        %   It is assumed that the quadrature nodes do not contain 0 as
        %   a frequency.
        %
        % INPUTS:
        %   obj      - instance of sampler class containing second-order
        %              system matrices and relevant dimensions
        %   nodesr   - quadrature nodes used in the implicit
        %              approximation
        %   weightsr - quadrature weights used in the implicit
        %              approximation
        %
        % OUTPUTS:
        %   Uquad -  approximate right quadrature-based square-root factor
        %           (n x m*J)
    
        % Last editied: 6/25/2024
        J = length(nodesr);
        Uquad = zeros(2*obj.n, J * obj.m);
        for j = 1:J
            V = weightsr(j)*obj.so_structured_solve(nodesr(j), 0, obj.time);
            Uquad(:, (j - 1) * obj.m + 1 : j * obj.m) = V;
        end
    end

    %%

    function Lquad = quad_left_factor(obj, nodesl, weightsl)
        % QUAD_LEFT_FACTOR Function to explicitly compute the
        % quadrature-based approximation to the frequency-limited
        % Gramian Q ~ Lquad*Lquad'
        %
        %
        % DESCRIPTION:
        %   Function computes multiple linear solves using the class 
        %   method `so_structure_solve' to compute the quadrature-based
        %   square root factor Lquad (n x p*K), K = length(nodesl).
        %   The kth (n x p) block of Lquad; is defined by:
        %
        %       W = conj(weightsl(k))*(conj(nodesl(k))*Efo' - Afo')\(Cfo')
        %
        %   Efo, Afo, and Cfo are the descriptor, mass, and output
        %   matrices obtained from the first-order realization of the
        %   underlying model. 
        %
        %   It is assumed that the quadrature nodes do not contain 0 as
        %   a frequency.
        %
        % INPUTS:
        %   obj      - instance of sampler class containing second-order
        %              system matrices and relevant dimensions
        %   nodesl   - quadrature nodes used in the implicit
        %              approximation
        %   weightsl - quadrature weights used in the implicit
        %              approximation
        %
        % OUTPUTS:
        %   Lquad -  approximate left quadrature-based square-root factor
        %           (n x p*K)
    
        % Last editied: 6/25/2024
        K = length(nodesl);
        Lquad = zeros(2*obj.n, K * obj.p);
        for k = 1:K
            % Conjugate of node included in `so_structured_solve' so don't
            % take conjugate here.
            W = conj(weightsl(k))*obj.so_structured_solve(nodesl(k), 1, obj.time);
            Lquad(:, (k - 1)*obj.p + 1:k*obj.p) = W;
        end
    end

    %%

    function [Efo, Afo, Bfo, Cfo, Dfo] = build_fo_realization(obj)
        % BUILD_FO_REALIZATION Function to build sparse first-order system
        % realizationfrom second-order matrices (for testing)
        % 

        % Build first-order realization from second-order parameters
        Efo                                   = spalloc(2*obj.n, 2*obj.n, nnz(obj.Mso) + obj.n); % Descriptor matrix; Efo = [I, 0: 0, Mso]
        Efo(1:obj.n, 1:obj.n)                 = speye(obj.n); % (1, 1) block
        Efo(obj.n+1:2*obj.n, obj.n+1:2*obj.n) = obj.Mso;      % (2, 2) block is (sparse) mass matrix
        
        Afo                                   = spalloc(2*obj.n, 2*obj.n, nnz(obj.Kso) + nnz(obj.Dso) + obj.n); % Afo = [0, I; -Kso, -Dso]
        Afo(1:obj.n, obj.n+1:2*obj.n)         = speye(obj.n); % (1, 2) block of Afo
        Afo(obj.n+1:2*obj.n, 1:obj.n)         = -obj.Kso;     % (2, 1) block is -Kso
        Afo(obj.n+1:2*obj.n, obj.n+1:2*obj.n) = -obj.Dso;     % (2, 2) block is -Dso
        
        Bfo             = spalloc(2*obj.n, obj.m, nnz(obj.Bso)); % Bfo = [0; Bso];
        Bfo(obj.n+1:2*obj.n, :) = obj.Bso; 
        
        % Position input, only
        Cfo             = spalloc(obj.p, 2*obj.n, nnz(obj.Cso)); % Cfo = [soC, 0];
        Cfo(:, 1:obj.n) = obj.Cso; 
        
        % No input-to-output term
        Dfo = zeros(obj.p, obj.m);
    end


    %%
    %     __                        
    %    (_   _. ._ _  ._  |  _   _ 
    %    __) (_| | | | |_) | (/_ _> 
    %                  |           
    %%

    function Gs = sample_G(obj, s)
        % SAMPLE_G Function to evaluate the transfer function of the
        % underlying linear model at prescribed frequencies.
        %
        % DESCRIPTION: 
        %   Evaluates the underlying transfer function at the
        %   prescribed frequencies in s. Each evaluation is computed 
        %   (in terms of the first-order realization) as
        %
        %       G(s(j)) - Dfo = Cfo*((s(j)*Efo - Afo)\Bfo) (1)
        %
        %   The the class method `so_structured_solve' is used to 
        %   compute the linear solve in (1)
        %
        % INPUTS:
        %   obj - instance of sampler class containing second-order
        %         system matrices and relevant dimensions
        %   s   - limited frequencies to evaluate at
        %
        % OUTPUTS:
        %   Gs -  transfer function evaluations (each of dim. (p x m))
        %         stored as (p x m x len(s)) array
        % Last editied: 6/25/2024

        % Define first-order output matrix for use in evaluating G
        % Position input, only
        Cfo             = spalloc(obj.p, 2*obj.n, nnz(obj.Cso)); % Cfo = [Cso, 0];
        Cfo(:, 1:obj.n) = obj.Cso; 

        % Space allocation
        Gs = zeros(obj.p, obj.m, length(s));
        for i = 1:length(s)
            V           = obj.so_structured_solve(s(i), 0, obj.time);
            Gs(:, :, i) = Cfo * V;
        end
    end

    %%

    function Gs = samples_for_Cr(obj, s)
        % SAMPLES_FOR_CR Function to generate samples used to fill
        % out the Loewner matrix Gbar that is used to compute the
        % reduced-order Cfo (first-order output matrix) in flQuadBT
        %
        % DESCRIPTION: 
        %   Call to `sample_G'. See `help.flQuadBTSampler.sample_G' for
        %   details.

        % Last editied: 6/25/2024

        Gs = obj.sample_G(s);
    end

    function Gs = samples_for_Ar_Er(obj, s)
        % SAMPLES_FOR_AR_Er Function to generate samples used to fill
        % out the Loewner matrices Lbar, Mbar, that are used to compute 
        % the reduced-order Afo, Efo (first-order mass and descriptor 
        % matrices) in flQuadBT
        %
        % DESCRIPTION: 
        %   Call to `sample_G'. See `help.flQuadBTSampler.sample_G' for
        %   details.

        % Last editied: 6/25/2024

        Gs = obj.sample_G(s);
    end

    function Gs = samples_for_Br(obj, s)
        % SAMPLES_FOR_BR Function to generate samples used to fill
        % out the Loewner matrix Hbar that is used to compute the
        % reduced-order Bfo (first-order input matrix) in flQuadBT
        %
        % DESCRIPTION: 
        %   Call to `sample_G'. See `help.flQuadBTSampler.sample_G' for
        %   details.

        % Last editied: 6/25/2024

        Gs = obj.sample_G(s);
    end

    end % End of methods
end % End of class 