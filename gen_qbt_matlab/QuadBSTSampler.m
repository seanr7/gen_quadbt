classdef QuadBSTSampler < GenericSampler & handle
    % Author: Sean Reiter (seanr7@vt.edu)

    % Class to generate relevant transfer function samples for use in Quadrature-Based Balanced Stochastic Truncation (QuadBST)

    % Paramters
    % ---------
    % A, B, C, D
    %   State-space realization of the FOM, matrices of size (n, n), (n, m), (p, n), and (p, m) respectively

    % For QuadBST, the relevant spectral factorization is that of 
    %   G(s) * G(-s).T = W(-s).T * W(s)

    properties 
        Dinv  
        %   Inverse of feedthrough; pre-computed for repeated use
        P
        %   `Left' Gramian to be balanced. 
        %       In QuadBST; P is the `controllability Gramian' of G(s), and solves;
        %       @math: `A * P + P * A' + B * B' = 0`
        Q     
        %   `Right' Gramian to be balanced. 
        %       In QuadBST; Q is the `observability Gramian' of W(s), and solves;
        %       @math: `A' * Q + Q * A + (C - B_W' * Q)'* (D * D')^{-1} * (C - B_W' * X) = 0`
        %              
        B_W   
        %   Input matrix of W(s)
        %       @math:  B_W := P * C' + B * D
        C_lsf 
        %   Output matrix of the relevant left spectral factor (lsf); W(s)
        %       @math: C_lsf := Dinv * (C - B_W' * Q)
        B_rsf 
        %   Input matrix of the relevant right spectral factor (rsf); In BST, this is just G(s)
        %       @math: B_rsf := B
    end
    
    methods
        function obj = QuadBSTSampler(A, B, C, D)
            % Call to Super
            obj@GenericSampler(A, B, C, D)
            % Two assumption:
            %   1. G(s) is square (m == p)
            if obj.m ~= obj.p
                error('G(s) is assumed to be a square system! (m == p)')
            end
            %   2. Feedthrough matrix (D) is nonsingular
            if any(rank(obj.D) ~= obj.m)
                error('D must be nonsingular!')
            end
            % Pre-compute Dinv
            obj.Dinv = obj.D \ eye(obj.m, obj.m);
        end

        %     __                          
        %    /__ ._ _. ._ _  o  _. ._   _ 
        %    \_| | (_| | | | | (_| | | _> 
        %                                 

        function P_ = get.P(obj)
            % Getter method for P
            % If not previously set, will solve for P 
            if isempty(obj.P)
                disp("Computing the reachability Gramian, P, of G(s)")
                % Solve the reachability ALE of G(s)
                %   @math: `A * P + P * A' + B * B' = 0`
                obj.P = lyap(obj.A, obj.B * obj.B');
            end
            P_ = obj.P;
        end

        function Q_ = get.Q(obj)
            % Getter method for Q
            % If not previously set, will solve for Q
            if isempty(obj.Q)
                disp("Computing the observability Gramian, Q, of W(s)")
                % Solve the observability ARE of W(s)
                %   @math: `A' * Q + Q * A + (C - B_W' * Q)'* (D * D')^{-1} * (C - B_W' * Q) = 0`
                % Fit the above to MATLAB's `icare(A, B, X, R, S, E, G)', that solves:
                %   A' * Q * E + E' * Q * A + E' * Q * G * Q * E - (E' * Q * B + S) * Rinv * (B' * Q * E + S') + X = 0
                [obj.Q, ~, ~] = icare(obj.A, -obj.B_W, zeros(obj.n, obj.n), -(obj.D * obj.D'), obj.C', obj.I, zeros(obj.n, obj.n));
            end
            Q_ = obj.Q;
        end

        function Utilde = right_sqrt_factor(obj, nodesr, weightsr)
            % Compute square-root factor used in approximating P ~ Utilde * Utilde';
            % For testing purposes
            J = length(nodesr); m = obj.m;
            Utilde = zeros(obj.n, J * m);
            for j = 1:J
                Utilde(:, (j - 1) * m + 1 : j * m) = weightsr(j) * ((nodesr(j) * obj.I - obj.A) \ obj.B_rsf);
            end
        end

        function Ltilde = left_sqrt_factor(obj, nodesl, weightsl)
            % Compute square-root factor used in approximating Q ~ Ltilde * Ltilde' (returns Ltilde)
            % For testing purposes
            K = length(nodesl); p = obj.p;
            Ltilde = zeros(obj.n, K * p);
            for k = 1:K
                Ltilde(:, (k - 1) * p + 1 : k * p) = conj(weightsl(k)) * ((conj(nodesl(k)) * obj.I - obj.A') \ obj.C_lsf');
            end
        end

        %     __                        
        %    (_   _. ._ _  ._  |  _   _ 
        %    __) (_| | | | |_) | (/_ _> 
        %                  |           

        function B_W_ = get.B_W(obj)
            % Getter method for B_W; input matrix of W(s)
            % Used in construction of C_lsf; unique to this class
            if isempty(obj.B_W)
                obj.B_W = obj.P * obj.C' + obj.B * obj.D;
            end
            B_W_ = obj.B_W;
        end

        function C_lsf_ = get.C_lsf(obj)
            % Getter method for C_lsf
            % Here, output matrix of W(s)
            %   @math: `C_lsf := C_W = D^{-1} * (C - B_W' * Q)`
            if isempty(obj.C_lsf)
                obj.C_lsf = obj.Dinv * (obj.C - obj.B_W' * obj.Q);
            end
            C_lsf_ = obj.C_lsf;
        end

        function B_rsf_ = get.B_rsf(obj)
            % Getter method for B_rsf
            % Here, rsf is G(s) itself, so return B
            B_rsf_ = obj.B;
        end

        function rsf_samples = samples_for_Gbar(obj, s)
            % Artificially sample the appropriate rsf to obtain samples for Gbar = C * Utilde;
            % Here, this is just G_infty(s), i.e. s.t.
            %   @math: `G(s) * G(-s).T = W(-s).T * W(s)`
            % Gbar used in building the reduced-order Cr
            rsf_samples = obj.sampleG(s); % Call to parent class
        end

        function lsf_samples = samples_for_Hbar(obj, s)
            % Artificially sample the appropriate lsf to obtain samples for Hbar = Ltilde' * B;
            % Here, this is the system cascade
            %   @math: `F(s) := [(W(-s).T^{-1}) * G(s)]_+ = C_lsf * (s * I - A) \ B`
            % Hbar used in building the reduced-order Brr

            % Pass to sampler for Lbar, Mbar, since required samples are the same
            lsf_samples = obj.samples_for_Lbar_Mbar(s);
        end

        function cascade_samples = samples_for_Lbar_Mbar(obj, s)
            % Artificially sample the appropriate system cascade to obtain samples for Lbar = Ltilde' * Utilde; Mbar = Ltilde' * A * Utilde;
            % Here, this is the system cascade
            %   @math: `F(s) := [(W(-s).T^{-1}) * G(s)]_+ = C_lsf * (s * I - A) \ B`
            % Lbar used in computing approximate singular values; Mbar used in building the reduced-order Ar

            % Will have length(s) samples of a (p x m) rational transfer matrix; space allocation
            cascade_samples = zeros(obj.p, obj.m, length(s));
            for i = 1:length(s)
                cascade_samples(:, :, i) = obj.C_lsf * ((s(i) * obj.I - obj.A) \ obj.B);
            end
        end
    end
end