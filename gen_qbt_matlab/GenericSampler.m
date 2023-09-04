classdef GenericSampler < handle
    properties
        A % State matrix
        B % Input matrix
        C % Output matrix
        D % Feedthrough
        n % State dimension
        m % Input dimension
        p % Output dimension
        I % E = I identity)
    end

    methods
        % Constructor
        function obj = GenericSampler(A, B, C, D)
            [obj.p, obj.m] = size(D);   [obj.n, ~] = size(A); % Save I-O and state dimension with class instance
            obj.A = A;  obj.B = B;  obj.C = C;  obj.D = D;
            obj.I = eye(obj.n, obj.n);
        end

        function Gs = sampleG(obj, s)
            % Function to artifically sample G_\infty(s) = G(s) - D
            Gs = zeros(obj.p, obj.m, length(s));
            for i = 1:length(s)
                Gs(:, :, i) = obj.C * ((s(i) * obj.I - obj.A) \ obj.B);
            end
        end
    end
end