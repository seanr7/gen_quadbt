classdef flQuadBTSampler < handle
% FLQUADBTSAMPLER Class to handle sample transfer function evaluations
% required in frequency-limited Quadrature-based Balanced Truncation 
% (flQuadBT)
%
% DESCRIPTION:
%   Computes transfer function evaluations of a linear time-invariant
%   system described by system matrices (E, A, B, C, D) for use in
%   flQuadBT
%
% PROPERTIES:
%   E   - first-order descriptor matrix (n x n)
%   A   - first-order mass matrix (n x n)
%   B   - first-order input matrix (n x m)
%   C   - first-order output matrix(p x n)  
%   n   - state dimension
%   m   - input dimension
%   p   - output dimension
%   typ - type of balancing beind done; `flbt'
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
        E
        A
        B
        C
        n
        m 
        p 
        typ
    end
    
    methods

    % Constructor method
    function obj = flQuadBTSampler(E, A, B, C, n, m, p)
        obj.E = E;  obj.A = A;  obj.B = B;  obj.C = C;

        % System matrices are sparse so state, input, output dimensions
        % are passed as arguments.
        obj.n = n;      obj.m = m;      obj.p = p;

        % Specify type of balancing so sampler class can play with reductor
        % class
        obj.typ = 'flbt';
    end

    %%
    %                              
    %   | | _|_ o | o _|_ o  _   _ 
    %   |_|  |_ | | |  |_ | (/_ _> 
    %          
    %%

    function Uquad = quad_right_factor(obj, nodesr, weightsr)
        % QUAD_RIGHT_FACTOR Function to explicitly compute the
        % quadrature-based approximation to the frequency-limited
        % Gramian P ~ Uquad*Uquad'
        %
        %
        % DESCRIPTION:
        %   Computes multiple linear for building quadrature-based
        %   square root factor Uquad (n x m*J), J = length(nodesr).
        %   The jth (n x m) block of Uquad is defined by:
        %
        %       V = weightsr(j)*(nodesr(j)*E - A)\B
        %
        %   E, A, and B are the descriptor, mass, and input matrices of the
        %   underling full order model.
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
        Uquad = zeros(obj.n, J * obj.m);
        for j = 1:J
            V = weightsr(j)*((nodesr(j)*obj.E - obj.A)\obj.B);
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
        %   Computes multiple linear for building quadrature-based
        %   square root factor Uquad (n x m*J), J = length(nodesr).
        %   The jth (n x m) block of Uquad is defined by:
        %
        %       W = conj(weightsl(k))*(conj(nodesl(k))*E' - A')\(C')
        %
        %   Efo, Afo, and Cfo are the descriptor, mass, and output
        %   matrices of the underlying full order model.
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
        Lquad = zeros(obj.n, K * obj.p);
        for k = 1:K
            % Conjugate of node included in `so_structured_solve' so don't
            % take conjugate here.
            W = conj(weightsl(k))*((conj(nodesl(k))*obj.E' - obj.A')\obj.C');
            Lquad(:, (k - 1)*obj.p + 1:k*obj.p) = W;
        end
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
        %   prescribed frequencies in s. Each evaluation is computed as
        %
        %       G(s(j)) - D = C*((s(j)*E - A)\B) (1)
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

        % Space allocation
        Gs = zeros(obj.p, obj.m, length(s));
        for i = 1:length(s)
            Gs(:, :, i) =obj.C*((s(i)*obj.E - obj.A)\obj.B);
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