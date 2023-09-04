classdef QuadPRBTSampler < GenericSampler & handle
    % Author: Sean Reiter (seanr7@vt.edu)

    % Class to generate relevant transfer function samples for use in Quadrature-Based Positive-real Balanced Truncation (QuadPRBT)

    % Paramters
    % ---------
    % A, B, C, D
    %   State-space realization of the FOM, matrices of size (n, n), (n, m), (p, n), and (p, m) respectively

    % For QuadPRBT, the relevant spectral factorization is that of the Popov function
    %   @math: `\phi(s) = G(s) + G(-s).T = M(-s).T * M(s) = N(s) * N(-s).T

    properties 
        R 
        %   @math: `R = D + D'`, 
        Rinv
        Rinv_sqrt
        %   Inverse of R and its matrix square-root; pre-computed for repeated use
        P
        %   `Left' Gramian to be balanced. 
        %       In QuadPRBT; P is the `controllability Gramian' of N(s), and solves;
        %       @math: `A * P + P * A' + (B - P * C')* (D + D')^{-1} * (B - C' * P)' = 0`
        Q     
        %   `Right' Gramian to be balanced. 
        %       In QuadPRBT; Q is the `observability Gramian' of M(s), and solves;
        %       @math: `A' * Q + Q * A + (C - B' * Q)'* (D + D')^{-1} * (C - B' * Q) = 0`
        %              
        C_lsf 
        %   Output matrix of the relevant left spectral factor (lsf); M(s)
        %       @math: C_lsf := Rinv_sqrt * (C - B' * Q)
        B_rsf 
        %   Input matrix of the relevant right spectral factor (rsf); N(s)
        %       @math: B_rsf := (B - P * C') * Rinv_sqrt
    end
    
    methods
        function obj = QuadPRBTSampler(A, B, C, D)
            % Call to Super
            obj@GenericSampler(A, B, C, D)
            % Two assumption:
            %   1. G(s) is square (m == p)
            obj.R = D + D';
            if obj.m ~= obj.p
                error('G(s) is assumed to be a square system! (m == p)')
            end
            %   2. System is positive-real (check that R = D + D.T > 0)
            if any(eig(R) < 0)
                error('G(s) is assumed to be positve-real!')
            end
            % Pre-compute Dinv
            obj.Rinv = obj.R \ eye(obj.m, obj.m);
            obj.Rinv_sqrt = sqrtm(obj.R);
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
                %   A * P + P * A' + (B - P * C') * R^{-1} * (B - P * C')' = 0
                % Fit the above to MATLAB's `icare(A, B, X, R, S, E, G)', that solves:
                %   A' * P * E + E' * P * A + E' * P * G * P * E - (E' * P * B + S) * Rinv * (B' * P * E + S') + X = 0
                [obj.P, ~, ~] = icare(obj.A', -obj.C', zeros(obj.n, obj.n), -obj.R, obj.B, obj.I, zeros(obj.n, obj.n));
            end
            P_ = obj.P;
        end

        function Q_ = get.Q(obj)
            % Getter method for Q
            % If not previously set, will solve for Q
            if isempty(obj.Q)
                disp("Computing the observability Gramian, Q, of M(s)")
                % Solve the observability ARE of M(s)
                %   A' * Q + Q * A + (C - B' * Q)'* R^{-1} * (C - B' * Q) = 0
                % Fit the above to MATLAB's `icare(A, B, X, R, S, E, G)', that solves:
                %   A' * Q * E + E' * Q * A + E' * Q * G * Q * E - (E' * Q * B + S)* Rinv *(B' * X * E + S') + X = 0
                [obj.Q, ~, ~] = icare(obj.A, -obj.B, zeros(obj.n, obj.n), -obj.R, obj.C', obj.I, zeros(obj.n, obj.n));
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

        function C_lsf_ = get.C_lsf(obj)
            % Getter method for C_lsf
            % Here, output matrix of M(s)
            %   C_lsf := C_M = R^{-1/2} * (C - B' * Q)
            if isempty(obj.C_lsf)
                obj.C_lsf = obj.Rinv_sqrt * (obj.C - obj.B' * obj.Q);
            end
            C_lsf_ = obj.C_lsf;
        end

        function B_rsf_ = get.B_rsf(obj)
            % Getter method for B_rsf
            % Here, input matrix of N(s)
            %   B_rsf := B_N = (B - P * C') * R^{-1/2}
            if isempty(obj.B_rsf)
                obj.B_rsf = (obj.B - obj.P * obj.C') * obj.Rinv_sqrt;
            end
            B_rsf_ = obj.B_rsf;
        end

        function rsf_samples = samples_for_Gbar(obj, s)
            % Artificially sample the appropriate rsf to obtain samples for Gbar = C * Utilde;
            % Here, this is the rsf of the Popov function
            %   @math: `N_\infty(s) = C * (s * I - A) \ B_N, where G(s) + G(-s).T = N(s) * N(-s).T`
            % Gbar used in building the reduced-order Cr

            % Space allocation; will have length(s) samples of a (p x m) rational transfer matrix
            rsf_samples = zeros(obj.p, obj.m, length(s)); 
            for i = 1:length(s)
                rsf_samples(:, :, i) = obj.C * ((s(i) * obj.I - obj.A) \ obj.B_rsf);
            end
        end

        function lsf_samples = samples_for_Hbar(obj, s)
            % Artificially sample the appropriate lsf to obtain samples for Hbar = Ltilde' * B;
            % Here, this is the lsf of the Popov function
            %   @math: `M_\infty(s) = C_M * (s * I - A) \ B, where G(s) + G(-s).T = M(-s).T * M(s)`
            % Hbar used in building the reduced-order Br

            % Space allocation; will have length(s) samples of a (p x m) rational transfer matrix
            lsf_samples = zeros(obj.p, obj.m, length(s)); 
            for i = 1:length(s)
                lsf_samples(:, :, i) = obj.C_lsf * ((s(i) * obj.I - obj.A) \ obj.B);
            end
        end

        function cascade_samples = samples_for_Lbar_Mbar(obj, s)
            % Artificially sample the appropriate system cascade to obtain samples for Lbar = Ltilde' * Utilde; Mbar = Ltilde' * A * Utilde;
            % Here, this is the system cascade
            %   F(s) := [(M(-s).T^{-1}) * N(s)]_+ = C_M * (s * I - A) \ B_N
            % Lbar used in computing approximate singular values; Mbar used in building the reduced-order Ar

              % Space allocation; will have length(s) samples of a (p x m) rational transfer matrix
              cascade_samples = zeros(obj.p, obj.m, length(s)); 
              for i = 1:length(s)
                  cascade_samples(:, :, i) = obj.C_lsf * ((s(i) * obj.I - obj.A) \ obj.B_rsf);
              end
        end
    end
end