classdef QuadBRBTSampler < GenericSampler & handle
    % Author: Sean Reiter (seanr7@vt.edu)

    % Class to generate relevant transfer function samples for use in Quadrature-Based Bounded-real Balanced Truncation (QuadBRBT)

    % Paramters
    % ---------
    % A, B, C, D
    %   State-space realization of the FOM, matrices of size (n, n), (n, m), (p, n), and (p, m) respectively

    % For QuadBRBT, the relevant spectral factorizations are
    %   @math: `I_p - G(s) * G(-s).T = N(s) * N(-s).T
    %   @math: `I_m - G(-s).T * G(s) = M(-s).T * M(s)

    properties 
        R_B 
        %   @math: `R_B = I_p - D * D'`
        R_Binv
        R_Binv_sqrt
        %   Inverse of R_B and its matrix square-root; pre-computed for repeated use
        R_C
        %   @math: `R_C = I_m - D' * D`
        R_Cinv
        R_Cinv_sqrt
        %   Inverse of R_C and its matrix square-root; pre-computed for repeated use
        P
        %   `Left' Gramian to be balanced. 
        %       In QuadBRBT; P is the `controllability Gramian' of N(s), and solves;
        %       @math: `A * P + P * A' + B * B' + (P * C' + B * D') * (I_p - D * D')^{-1} * (P * C' + B * D')' = 0`
        Q     
        %   `Right' Gramian to be balanced. 
        %       In QuadBRBT; Q is the `observability Gramian' of M(s), and solves;
        %       @math: `A' * Q + Q * A + C' * C + (B' * Q + D' * C)' * (I_m - D' * D)^{-1} * (B' * Q + D' * C) = 0`
        %              
        C_lsf 
        %   Output matrix of the relevant left spectral factor (lsf); M(s)
        %       @math: C_lsf := -R_Cinv_sqrt * (B' * Q + D * C)
        B_rsf 
        %   Input matrix of the relevant right spectral factor (rsf); N(s)
        %       @math: B_rsf := -(P * C' + B * D') * R_Binv_sqrt
    end
    
    methods
        function obj = QuadBRBTSampler(A, B, C, D)
            % Call to Super
            obj@GenericSampler(A, B, C, D)
            % Two assumption:
            %   1. G(s) is bounded-real (I - D' * D > 0)
            obj.R_C = eye(obj.m, obj.m) - (obj.D' * obj.D);
            if any(eig(obj.R_C) < 0) % IO dimensions are small, so computing this doesn't hurt
                error('G(s) is assumed to be bounded-real!')
            end
            %   2. Dual of G(s) is bounded-real (I - D * D' > 0)
            obj.R_B = eye(obj.p, obj.p) - (obj.D * obj.D');
            if any(eig(obj.R_B) < 0) % IO dimensions are small, so computing this doesn't hurt
                error('G(s) is assumed to be bounded-real!')
            end
            % Pre-compute 
            obj.R_Cinv = obj.R_C \ eye(obj.m, obj.m);
            obj.R_Cinv_sqrt = chol(obj.R_Cinv);
            obj.R_Binv = obj.R_B \ eye(obj.p, obj.p);
            obj.R_Binv_sqrt = chol(obj.R_Binv);
        end

        %     __                          
        %    /__ ._ _. ._ _  o  _. ._   _ 
        %    \_| | (_| | | | | (_| | | _> 
        %                                 

        function P_ = get.P(obj)
            % Getter method for P
            % If not previously set, will solve for P 
            if isempty(obj.P)
                disp("Computing the reachability Gramian, P, of N(s)")
                % Solve the reachability ARE of N(s)
                %   @math: `A * P + P * A' + B * B' + (P * C' + B * D') * R_B^{-1} * (P * C' + B * D')' = 0`
                % Fit the above to MATLAB's `icare(A, B, X, R, S, E, G)', that solves:
                %   A' * P * E + E' * P * A + E' * P * G * P * E - (E' * P * B + S) * Rinv * (B' * P * E + S') + X = 0
                [obj.P, ~, ~] = icare(obj.A', obj.C', (obj.B * obj.B'), -obj.R_B, (obj.B * obj.D'), obj.I, zeros(obj.n, obj.n));
            end
            P_ = obj.P;
        end

        function Q_ = get.Q(obj)
            % Getter method for Q
            % If not previously set, will solve for Q
            if isempty(obj.Q)
                disp("Computing the observability Gramian, Q, of M(s)")
                % Solve the observability ARE of M(s)
                %   @math: `A' * Q + Q * A + C' * C + (B' * Q + D' * C)' * (I_m - D' * D)^{-1} * (B' * Q + D' * C) = 0`
                % Fit the above to MATLAB's `icare(A, B, X, R, S, E, G)', that solves:
                %   A' * Q * E + E' * Q * A + E' * Q * G * Q * E - (E' * Q * B + S)* Rinv *(B' * X * E + S') + X = 0
                [obj.Q, ~, ~] = icare(obj.A, obj.B, (obj.C' * obj.C), -obj.R_C, (obj.C' * obj.D), obj.I, zeros(obj.n, obj.n));
            end
            Q_ = obj.Q;
        end

        function Utilde = right_sqrt_factor(obj, nodesr, weightsr)
            % Compute square-root factor used in approximating P ~ Utilde * Utilde';
            % For testing purposes
            J = length(nodesr); m = obj.m;
            % Column dimension is J * (2 * m), since B_rsf := [B, B_N]
            Utilde = zeros(obj.n, J * (2 * m));
            for j = 1:J
                Utilde(:, (j - 1) * (2 * m) + 1 : j * (2 * m)) = weightsr(j) * ((nodesr(j) * obj.I - obj.A) \ obj.B_rsf);
            end
        end

        function Ltilde = left_sqrt_factor(obj, nodesl, weightsl)
            % Compute square-root factor used in approximating Q ~ Ltilde * Ltilde' (returns Ltilde)
            % For testing purposes
            K = length(nodesl); p = obj.p;
            % Column dimension is K * (2 * p), since C_lsf := [C; C_M]
            Ltilde = zeros(obj.n, K * (2 * p));
            for k = 1:K
                Ltilde(:, (k - 1) * (2 * p) + 1 : k * (2 * p)) = conj(weightsl(k)) * ((conj(nodesl(k)) * obj.I - obj.A') \ obj.C_lsf');
            end
        end

        %     __                        
        %    (_   _. ._ _  ._  |  _   _ 
        %    __) (_| | | | |_) | (/_ _> 
        %                  |           

        function C_lsf_ = get.C_lsf(obj)
            % Getter method for C_lsf
            % Here, C_lsf := [C; C_M], where
            %   @math: `C_M := -R_C^{-1/2} * (B' * Q + D' * C)`
            if isempty(obj.C_lsf)
                obj.C_lsf = [obj.C; -obj.R_Cinv_sqrt * (obj.B' * obj.Q + obj.D' * obj.C)];
            end
            C_lsf_ = obj.C_lsf;
        end

        function B_rsf_ = get.B_rsf(obj)
            % Getter method for B_rsf
            % Here, B_rsf := [B, B_N], where
            %   @math: `B_N := -(P * C' + B * D') * R_B^{-1/2}`
            if isempty(obj.B_rsf)
                obj.B_rsf = [obj.B, -(obj.P * obj.C' + obj.B * obj.D') * obj.R_Binv_sqrt];
            end
            B_rsf_ = obj.B_rsf;
        end

        function rsf_samples = samples_for_Gbar(obj, s)
            % Artificially sample the appropriate rsf to obtain samples for Gbar = C * Utilde;
            % Here, this is given by
            %   @math: `[G_\infty(s), N_\infty(s)] = C * (s * I - A) \ [B, B_N]`
            % Gbar used in building the reduced-order Cr

            % Space allocation; will have length(s) samples of a (p x (2 * m)) rational transfer matrix
            rsf_samples = zeros(obj.p, 2 * obj.m, length(s)); 
            for i = 1:length(s)
                rsf_samples(:, :, i) = obj.C * ((s(i) * obj.I - obj.A) \ obj.B_rsf);
            end
        end

        function lsf_samples = samples_for_Hbar(obj, s)
            % Artificially sample the appropriate lsf to obtain samples for Hbar = Ltilde' * B;
            % Here, this is given by
            %   @math: `[G_\infty(s); M_\infty(s)] = [C; C_M] * (s * I - A) \ B`
            % Hbar used in building the reduced-order Br

            % Space allocation; will have length(s) samples of a ((2 * p) x m) rational transfer matrix
            lsf_samples = zeros(2 * obj.p, obj.m, length(s)); 
            for i = 1:length(s)
                lsf_samples(:, :, i) = obj.C_lsf * ((s(i) * obj.I - obj.A) \ obj.B);
            end
        end

        function cascade_samples = samples_for_Lbar_Mbar(obj, s)
            % Artificially sample the appropriate system cascade to obtain samples for Lbar = Ltilde' * Utilde; Mbar = Ltilde' * A * Utilde;
            % Here, this is given by
            %   @math: `[G_\infty(s), N_\infty(s); M_\infty(s), -((M(-s).T)^{-1} * D * N(s))] = [C; C_M] * (s * I - A) \ [B, B_N]
            % Lbar used in computing approximate singular values; Mbar used in building the reduced-order Ar

            % Space allocation; will have length(s) samples of a ((2 * p) x (2 * m)) rational transfer matrix
            cascade_samples = zeros(2 * obj.p, 2 * obj.m, length(s)); 
            for i = 1:length(s)
                cascade_samples(:, :, i) = obj.C_lsf * ((s(i) * obj.I - obj.A) \ obj.B_rsf);
            end
        end
    end
end