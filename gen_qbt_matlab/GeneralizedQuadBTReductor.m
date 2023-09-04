classdef GeneralizedQuadBTReductor < handle
    % Author: Sean Reiter (seanr7@vt.edu)
    
    % Generalized Quadrature-based Balanced Truncation (GenQuadBT) Reductor class;
    % QuadBT was originally presented in :cite: `gosea2022data`
    
    % This code handles a generalized version that extends to Positive-real Balanced Truncation (PRBT) :cite: `desai1984transformation`,
    % Balanced Stochastic Truncation (BST) :cite: `desai1984transformation`, Bounded-real Balanced Truncation (BRBT) :cite: `opdenacker1988contraction`
    
    properties
        sampler
        %   Child of |GenericSampler| class to synthetically generate transfer function `data` relevant to the approximate balancing being performed
        nodesl
        %   `Left` interpolation points (quadrature modes) as a (K) dim array.
        %   Implicitly quadrature modes used in approximating the `observability` Gramian Q relevant to the underlying balancing
        %       Use `left` designation because these points are used in the approximate `left` quadrature-based square-root factor L; Q ~ L * L.T
        nodesr
        %   `Right` interpolation point (quadrature modes) as a (J) dim array.
        %   Implicitly quadrature modes used in approximating the `reachability` Gramian P relevant to the underlying balancing
        %       Use `right` designation because these points are used in the approximate `right` quadrature-based square-root factor U; P ~ U * U.T
        weightsl
        %   `Left` quadrature weights as an (K) dim array.
        weightsr
        %   `Right` quadrature weights as a (J) dim array.
        Lbar % @math: `:= Ltilde' * Utilde`
        Mbar % @math: `:= Ltilde' * A * Utilde`
        Hbar % @math: `:= Ltilde' * B`
        Gbar % @math: `:= C * Utilde`
        %   Loewner quadruple (Lbar, Mbar, Hbar, Gbar); used in approximating (Hsv, Arbar, Brbar, Crbar), respectively
        Zbar
        Sbar
        Ybar
        hsvbar
        %   Approximate svd `from data'; i.e.
        %       @math: `Lbar = Zbar * Sbar * Ybar';    hsvbar = diag(Sbar)`
        %       Economy-sized svd of Lbar
    end

    methods
        function obj = GeneralizedQuadBTReductor(sampler, nodesl, nodesr, weightsl, weightsr)
            % Save modes / weights with instance of class
            obj.nodesl = nodesl;        obj.nodesr = nodesr;
            obj.weightsl = weightsl;    obj.weightsr = weightsr; 
            % Instance of `sampler' class for generating relevant samples 
            obj.sampler = sampler;
        end

        %                                  _
        %    |   _   _       ._   _  ._   |_ ._   _  o ._   _
        %    |_ (_) (/_ \/\/ | | (/_ |    |_ | | (_| | | | (/_
        %                                         _|

        % Methods for building relevant Loewner matrices
        % Loewner quadrauple (Lbar, Mbar, Hbar, Gbar) is built once then cached with instance of class
        % Can then be used for computing RoMs of different orders r

        function Lbar_ = get.Lbar(obj)
            if isempty(obj.Lbar)
                % If not yet computed; Call to static method, save with instance of class
                disp("Computing Lbar")
                % Compute left, right samples from sampler class
                % Samples (Gsl, Gsr) have dimensions (p x m x K) and (p x m x J), respectively
                Gsl = obj.sampler.samples_for_Lbar_Mbar(obj.nodesl);    
                Gsr = obj.sampler.samples_for_Lbar_Mbar(obj.nodesr);
                obj.Lbar = obj.Lbar_(obj.nodesl, obj.nodesr, Gsl, Gsr, obj.weightsl, obj.weightsr);
            end
            Lbar_ = obj.Lbar;
        end

        function Mbar_ = get.Mbar(obj)
            if isempty(obj.Mbar)
                % If not yet computed; Call to static method, save with instance of class
                disp("Computing Mbar")
                % Compute left, right samples from sampler class
                % Samples (Gsl, Gsr) have dimensions (p x m x K) and (p x m x J), respectively
                Gsl = obj.sampler.samples_for_Lbar_Mbar(obj.nodesl); 
                Gsr = obj.sampler.samples_for_Lbar_Mbar(obj.nodesr);
                obj.Mbar = obj.Mbar_(obj.nodesl, obj.nodesr, Gsl, Gsr, obj.weightsl, obj.weightsr);
            end
            Mbar_ = obj.Mbar;
        end

        function Hbar_ = get.Hbar(obj)
            if isempty(obj.Hbar)
                % Call static method
                disp("Computing Hbar")
                % Compute left samples from sampler class
                Gsl = obj.sampler.samples_for_Hbar(obj.nodesl);
                obj.Hbar = obj.Hbar_(obj.nodesl, Gsl, obj.weightsl);
            end
            Hbar_ = obj.Hbar;
        end

        function Gbar_ = get.Gbar(obj)
            if isempty(obj.Gbar)
                % Call static method
                disp("Computing Gbar")
                % Compute right samples from sampler class
                Gsr = obj.sampler.samples_for_Gbar(obj.nodesr);
                obj.Gbar = obj.Gbar_(obj.nodesr, Gsr, obj.weightsr);
            end
            Gbar_ = obj.Gbar;
        end

        %     _                            
        %    |_)  _   _|      _ _|_  _  ._ 
        %    | \ (/_ (_| |_| (_  |_ (_) |  
        %   

        % Functionality to compute singular value decomposition (SVD) of Lbar + perform reduction

        function [Zbar_, Sbar_, Ybar_] = svd_from_data(obj) 
            % Compute approximate SVD of Lbar = Ltilde' * Utilde
            % At first call, this saves the approximate svd of Lbar with instance of the class
            if isempty(obj.Zbar)
                disp('Computing the SVD of Lbar')
                [obj.Zbar, obj.Sbar, obj.Ybar] = svd(obj.Lbar, 'econ');
            end
            Zbar_ = obj.Zbar;   Sbar_ = obj.Sbar;  Ybar_ = obj.Ybar;
        end

        function hsvbar_ = get.hsvbar(obj)
            % Getter method for hsvbar
            if isempty(obj.hsvbar)
                % If we haven't computed the svd yet, hsvbar will be empty
                % So, compute the svd of Lbar
                [~, Sbar_, ~] = obj.svd_from_data;
                obj.hsvbar = diag(Sbar_); % Grab diagonal
            end
            hsvbar_ = obj.hsvbar;
        end

        function [Arbar, Brbar, Crbar] = reduce(obj, r)
            % Perform reduction; order 1 \leq r < nn
            % ROM via PG-projection to approximate SBT; Dr := D is unchanged in ROM
            [~, ~, ~] = obj.svd_from_data; % This computes approximate SVD, saves with class
            Arbar = obj.Sbar(1 : r, 1 : r)^(-1/2) * obj.Zbar(:, 1 : r)' * obj.Mbar * obj.Ybar(:, 1 : r) * obj.Sbar(1 : r, 1 : r)^(-1/2);
            Brbar = obj.Sbar(1 : r, 1 : r)^(-1/2) * obj.Zbar(:, 1 : r)' * obj.Hbar;
            Crbar = obj.Gbar * obj.Ybar(:, 1 : r) * obj.Sbar(1 : r, 1 : r)^(-1/2);
        end

    end                               

    methods(Static)
        function Lbar_ = Lbar_(nodesl, nodesr, Gsl, Gsr, weightsl, weightsr)
            % Build scaled Loewner matrix Lbar with entries defined by
            %   @math: `Lbar_k,j = -weightsl(k) * weightsr(j) * (Gsl(:, :, k) - Gsr(:, :, j)) ./ (nodesl(k) - nodesr(j))`

            % Lbar replaces the product of exact square-root factors (and thus its svs approximate the true hsvs) in QuadBT;
            %   @math: `Lbar = Ltilde' * Utilde`

            % Parameters
            % ----------
            % nodesl
            %   `Left` interpolation points (quadrature modes) as an (K, ) dim array.
            % nodesr
            %   `Right` interpolation point (quadrature modes) as an (J, ) dim array.
            % Gsl
            %   `Left` transfer function data as (p, m, K) array.
            % Gsr
            %   `Right` transfer function data as (p, m, J) array.
            % weightsl
            %   `Left` quadrature weights as an (K, ) dim array.
            % weightsr
            %   `Right` quadrature weights as an (J, ) dim array.

            % Returns
            % -------
            % Lbar_
            %   Scaled Loewner matrix as (K * p, J * m) array.

            % Assumptions
            % -----------
            % Gsl and Gsr are generated by the same transfer function

            % Get dimensions of Loewner matrix
            K = length(nodesl);    J = length(nodesr);  [p, m] = size(Gsl(:, :, 1));
            Lbar_ = zeros(K * p, J * m);
            for k = 1:K % Run through rows
                for j = 1:J % Run through columns
                    % Pre-compute denominator
                    diff = nodesl(k) - nodesr(j);
                    Lbar_((k - 1) * p + 1 : k * p, (j - 1) * m + 1 : j * m) = -weightsl(k) * weightsr(j) * ((Gsl(:, :, k) - Gsr(:, :, j)) / diff);
                end
            end
        end

        function Mbar_ = Mbar_(nodesl, nodesr, Gsl, Gsr, weightsl, weightsr)
            % Build scaled shifted-Loewner matrix Mbar with entries defined by
            %   @math: `Mbar_k,j = -weightsl(k) * weightsr(j) * (nodesl(k) * Gsl(:, :, k) - nodesr(j) * Gsr(:, :, j)) ./ (nodesl(k) - nodesr(j))`

            % Mbar is used in building the reduced Ar matrix in QuadBT;
            %   @math: `Mbar = Ltilde' * A * Utilde`

            % Parameters
            % ----------
            % nodesl
            %   `Left` interpolation points (quadrature modes) as an (K, ) dim array.
            % nodesr
            %   `Right` interpolation point (quadrature modes) as an (J, ) dim array.
            % Gsl
            %   `Left` transfer function data as (p, m, K) array.
            % Gsr
            %   `Right` transfer function data as (p, m, J) array.
            % weightsl
            %   `Left` quadrature weights as an (K, ) dim array.
            % weightsr
            %   `Right` quadrature weights as an (J, ) dim array.

            % Returns
            % -------
            % Mbar_
            %   Scaled shifted-Loewner matrix as (K * p, J * m) array

            % Assumptions
            % -----------
            % Gsl and Gsr are generated by the same transfer function

            % Dimensions and pre-allocation
            K = length(nodesl);    J = length(nodesr);  [p, m] = size(Gsl(:, :, 1));
            Mbar_ = zeros(K * p, J * m);
            for k = 1:K % Run through rows
                for j = 1:J % Run through columns
                    % Pre-compute denominator
                    diff = nodesl(k) - nodesr(j);
                    Mbar_((k - 1) * p + 1 : k * p, (j - 1) * m + 1 : j * m) = -weightsl(k) * weightsr(j) * ((nodesl(k) * Gsl(:, :, k) - nodesr(j) * Gsr(:, :, j)) / diff);
                end
            end
        end

        function Hbar_ = Hbar_(nodesl, Gsl, weightsl)
            % Build input matrix in Loewner quadruple,
            %   @math: `Hbar(k, :) = weightsl(k) * Gsl(:, :, k)`
            % Hbar is used in constructing the reduced Br matrix in QuadBT;
            %   @math: `Hbar = Ltilde' * B`

            % Parameters
            % ----------
            % nodesl
            %     `Left` interpolation points (quadrature modes) as an (K, ) dim array.
            % Gsl
            %     `Left` transfer function data as (p, m, K) array.
            % weightsl
            %     `Left` quadrature weights as an (K, ) dim array.

            % Returns
            % -------
            % Hbar_
            %   Scaled input matrix in Loewner quadruple as (K * p, m) array.

            % Dimensions and pre-allocation
            K = length(nodesl); [p, m] = size(Gsl(:, :, 1));
            Hbar_ = zeros(K * p, m);   
            for k = 1:K
                Hbar_((k - 1) * p + 1 : k * p, :) = weightsl(k) * Gsl(:, :, k);  
            end
        end

        function Gbar_ = Gbar_(nodesr, Gsr, weightsr)
            % Build output matrix in Loewner quadruple,
            %   @math: `Gbar(:, j) = weightsr(j) * Gsr(:, :, j)`
            % Gbar is used in constructing the reduced Cr matrix in QuadBT;
            %   @math: `Gbar = C * Utilde`

            % Parameters
            % ----------
            % nodesr
            %     `Right` interpolation points (quadrature modes) as an (J, ) dim array.
            % Gsr
            %     `Right` transfer function data as (p, m, J) array.
            % weightsr
            %     `Right` quadrature weights as an (J, ) dim array.

            % Returns
            % -------
            % Gbar_
            %   Scaled output matrix in Loewner quadruple as (p, J * m) array.

            % Dimensions and pre-allocation
            J = length(nodesr); [p, m] = size(Gsr(:, :, 1));
            Gbar_ = zeros(p, J * m);
            for j = 1:J
                Gbar_(:, (j - 1) * m + 1 : j * m) = weightsr(j) * Gsr(:, :, j);  
            end
        end
    end
end