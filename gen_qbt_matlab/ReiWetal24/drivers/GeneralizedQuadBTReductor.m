classdef GeneralizedQuadBTReductor < handle
    % GENERALIZEDQUADBTREDUCTOR General reductor class to handle produce
    % Quadrature-based Balanced Truncation (QuadBT) reduced models.
    %
    % DESCRIPTION:
    %   Given data, quadrature weights and nodes, an instance of the
    %   GeneralizedQuadBTReductor class handles all production of QuadBT
    %   reduced models. 
    %
    % PROPERTIES:
    %   sampler  - Instance of sampler class; used to generate transfer
    %              function data relevant to the type of balancing being
    %              performed
    %   nodesl   - `left' quadrature nodes (interpolation points) as 
    %              (K x 1) array; used in the implicit quadrature rule 
    %              underlying the relevant `observability Gramian'
    %   nodesr   - `right' quadrature nodes (interpolation points) as 
    %              (J x 1) array; used in the implicit quadrature rule 
    %              underlying the relevant `controllability Gramian'
    %   weightsl - `left' quadrature weights as (K x 1) array
    %   weightsr - `right' quadrature weights as (J x 1) array  
    %   Lbar     - Loewner matrix; used in non-intrusuve approximation of
    %              reduced Er (and, approximate Hankel singular value
    %              triplet); Er = Lquad' * E * Uquad
    %   Mbar     - Loewner matrix; used in non-intrusuve approximation of
    %              reduced Ar; Ar = Lquad' * A * Uquad
    %   Hbar     - Loewner matrix; used in non-intrusuve approximation of
    %              reduced Br; Br = Lquad' * B
    %   Gbar     - Loewner matrix; used in non-intrusuve approximation of
    %              reduced Cr; Er = C * Uquad
    %   Zbar     - Left singular vectors of Lbar
    %   Sbar     - Singular values of Lbar
    %   Ybar     - Right singular vectors of Lbar
    %   hsvbar   - Hankel singular values from data; hsvbar = diag(Sbar)
    %   typ      - type of balancing beind done; inherited from obj.sampler
    %
    
    %
    % This file is part of the archive Code, Data, and Results for 
    % Numerical Experiments in "..."
    % Copyright (c) 2024 Sean Reiter, Steffen W. R. Werner
    % All rights reserved.
    % License: BSD 2-Clause license (see COPYING)
    %
    
    % Virginia Tech, Department of Mathematics
    % Last editied: 6/25/2024
    % Author: Sean Reiter (seanr7@vt.edu)
    
    % See class description for details of properties.
    properties  
        sampler
        nodesl
        nodesr
        weightsl
        weightsr
        Lbar 
        Mbar 
        Hbar 
        Gbar 
        Zbar
        Sbar
        Ybar
        hsvbar
        typ
    end

    methods

        % Constructor method
        function obj = GeneralizedQuadBTReductor(sampler, nodesl, nodesr, weightsl, weightsr)
            % Quadrature weights and nodes implicit in approximating the
            % relevant system Gramians
            obj.nodesl   = nodesl;      obj.nodesr   = nodesr;
            obj.weightsl = weightsl;    obj.weightsr = weightsr; 

            % Pass instance of sampler class to generate relevant data
            obj.sampler = sampler;
            obj.typ     = sampler.typ; % Type of balancing inherited from sampler class
            
        end

        %%
        %                                  _
        %    |   _   _       ._   _  ._   |_ ._   _  o ._   _
        %    |_ (_) (/_ \/\/ | | (/_ |    |_ | | (_| | | | (/_
        %                                         _|
        %
        %%

        % Factory methods for computing Loewner quadruple (Lbar, Mbar, Hbar, Gbar)
        % Note: These matrices are built once at first call, and then
        %       cached with class instance. This allows for recycling to
        %       compute reduced models for multiple orders of
        %       approximation, r

        function Lbar_ = get.Lbar(obj)
            % LBAR Getter method for Lbar
            %
            %
            % DESCRIPTION:
            %   At first call, calls obj.sampler class methods to
            %   artificially produce necessary transfer function data, and
            %   subsequently calls static method obj.Lbar_ to compute 
            %   Loewner matrix Lbar, and cache with class instance.
            %
            %   At later calls, gets cached Lbar.
            %
            % INPUTS:
            %   obj - class instance
            %
            % OUTPUTS:
            %   Lbar_ - Loewner matrix Lbar, computed using obj.Lbar_ (see
            %           help.GeneralizedQuadBTReductor.Lbar_ for details).
            %

            % Last editied: 6/25/2024

            % At first call, compute Loewner matrix Lbar with call to
            % static method obj.Lbar_
            if isempty(obj.Lbar)
                Lbar_start = tic;
                fprintf(1, 'CONSTRUCTING Lbar.\n')
                fprintf(1, '------------------\n')
                % 1. Call to instance of sampler class to compute the
                % required transfer function data for the tye of balancing
                % being performed.

                % Samples (Gsl, Gsr) have dimensions (p x m x K) and 
                % (p x m x J), respectively
                if strcmp(obj.typ, 'fwbt')
                    % If freqency-weighted BT; specialized call to Loewner 
                    % constuctors
                    [Wosl, Gsl, Wisr] = obj.sampler.samples_for_Ar_Er(obj.nodesl, ...
                        obj.nodesl, obj.nodesr);    
                    Gsr               = obj.sampler.sampleG(obj.nodesr);
                    obj.Lbar          = obj.Lbar_fwbt_(obj.nodesl, obj.nodesr, ...
                        Gsl, Gsr, Wosl, Wisr, obj.weightsl, obj.weightsr);
                else
                    % Otherwise, standard calls
                    Gsl      = obj.sampler.samples_for_Ar_Er(obj.nodesl);    
                    Gsr      = obj.sampler.samples_for_Ar_Er(obj.nodesr);
                    obj.Lbar = obj.Lbar_(obj.nodesl, obj.nodesr, Gsl, Gsr, ...
                        obj.weightsl, obj.weightsr);
                end
                fprintf(1, 'CONSTRUCTION OF Lbar FINISHED IN %.2f s\n', toc(Lbar_start))
                fprintf(1, '-----------------------------------------\n')
            end
            % Lbar now saved with instance of class, get
            Lbar_ = obj.Lbar;
        end

        function Mbar_ = get.Mbar(obj)            
            % MBAR Getter method for Mbar
            %
            %
            % DESCRIPTION:
            %   At first call, calls obj.sampler class methods to
            %   artificially produce necessary transfer function data, and
            %   subsequently calls static method obj.Mbar_ to compute 
            %   Loewner matrix Lbar, and cache with class instance.
            %
            %   At later calls, gets cached Mbar.
            %
            % INPUTS:
            %   obj - class instance
            %
            % OUTPUTS:
            %   Mbar_ - Loewner matrix Lbar, computed using obj.Mbar_ (see
            %           help.GeneralizedQuadBTReductor.Mbar_ for details).
            %

            % Last editied: 6/25/2024

            % At first call, compute Loewner matrix Lbar with call to
            % static method obj.Mbar_
            if isempty(obj.Mbar)
                Mbar_start = tic;
                fprintf(1, 'CONSTRUCTING Mbar.\n')
                fprintf(1, '------------------\n')
                % 1. Call to instance of sampler class to compute the
                % required transfer function data for the tye of balancing
                % being performed.
                
                % Samples (Gsl, Gsr) have dimensions (p x m x K) and 
                % (p x m x J), respectively
                if strcmp(obj.typ, 'fwbt')
                    % If freqency-weighted BT; specialized call to Loewner 
                    % constuctors
                    [Wosl, Gsl, Wisr] = obj.sampler.samples_for_Ar_Er(obj.nodesl, ...
                        obj.nodesl, obj.nodesr);    
                    Gsr              = obj.sampler.sampleG(obj.nodesr);
                    obj.Mbar         = obj.Mbar_fwbt_(obj.nodesl, obj.nodesr, ...
                        Gsl, Gsr, Wosl, Wisr, obj.weightsl, obj.weightsr);
                else
                    Gsl      = obj.sampler.samples_for_Ar_Er(obj.nodesl);    
                    Gsr      = obj.sampler.samples_for_Ar_Er(obj.nodesr);
                    obj.Mbar = obj.Mbar_(obj.nodesl, obj.nodesr, Gsl, Gsr, ...
                        obj.weightsl, obj.weightsr);
                end
                fprintf(1, 'CONSTRUCTION OF Mbar FINISHED IN %.2f s\n', toc(Mbar_start))
                fprintf(1, '-----------------------------------------\n')
            end
            % Mbar now saved with instance of class, get
            Mbar_ = obj.Mbar;
        end

        function Hbar_ = get.Hbar(obj)
            % HBAR Getter method for Hbar
            %
            %
            % DESCRIPTION:
            %   At first call, calls obj.sampler class methods to
            %   artificially produce necessary transfer function data, and
            %   subsequently calls static method obj.Hbar_ to compute 
            %   Loewner matrix Lbar, and cache with class instance.
            %
            %   At later calls, gets cached Hbar.
            %
            % INPUTS:
            %   obj - class instance
            %
            % OUTPUTS:
            %   Hbar_ - Loewner matrix Hbar, computed using obj.Hbar_ (see
            %           help.GeneralizedQuadBTReductor.Hbar_ for details).
            %

            % Last editied: 6/25/2024
                        
            % At first call, compute Loewner matrix Lbar with call to
            % static method obj.Hbar_
            if isempty(obj.Hbar)
                Hbar_start = tic;
                fprintf(1, 'CONSTRUCTING Hbar.\n')
                fprintf(1, '------------------\n')
                Gsl      = obj.sampler.samples_for_Br(obj.nodesl);
                obj.Hbar = obj.Hbar_(obj.nodesl, Gsl, obj.weightsl);
                fprintf(1, 'CONSTRUCTION OF Hbar FINISHED IN %.2f s\n', toc(Hbar_start))
                fprintf(1, '-----------------------------------------\n')
            end
            % Hbar now saved with instance of class, get
            Hbar_ = obj.Hbar;
        end

        function Gbar_ = get.Gbar(obj)
            % GBAR Getter method for Gbar
            %
            %
            % DESCRIPTION:
            %   At first call, calls obj.sampler class methods to
            %   artificially produce necessary transfer function data, and
            %   subsequently calls static method obj.Gbar_ to compute 
            %   Loewner matrix Gbar, and cache with class instance.
            %
            %   At later calls, gets cached Gbar.
            %
            % INPUTS:
            %   obj - class instance
            %
            % OUTPUTS:
            %   Gbar_ - Loewner matrix Gbar, computed using obj.Gbar_ (see
            %           help.GeneralizedQuadBTReductor.Gbar_ for details).
            %

            % Last editied: 6/25/2024

            % At first call, compute Loewner matrix Lbar with call to
            % static method obj.Gbar_
            if isempty(obj.Gbar)
                Gbar_start = tic;
                fprintf(1, 'CONSTRUCTING Gbar.\n')
                fprintf(1, '------------------\n')
                % Compute relevant data
                Gsr      = obj.sampler.samples_for_Cr(obj.nodesr);
                obj.Gbar = obj.Gbar_(obj.nodesr, Gsr, obj.weightsr);
                fprintf(1, 'CONSTRUCTION OF Gbar FINISHED IN %.2f s\n', toc(Gbar_start))
                fprintf(1, '-----------------------------------------\n')
            end
            % Gbar now saved with instance of class, get
            Gbar_ = obj.Gbar;
        end

        %%
        %     _                            
        %    |_)  _   _|      _ _|_  _  ._ 
        %    | \ (/_ (_| |_| (_  |_ (_) |  
        % 
        %%

        % Factory methods for computing Singular Value Decomposition 
        % triplet of obj.Lbar: (Zbar, Sbar, Ybar)
        % Note: These matrices are built once at first call, and then
        %       cached with class instance. This allows for recycling to
        %       compute reduced models for multiple orders of
        %       approximation, r

        % Here in cleaning of code - SR

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
    
    % Static methods
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
                    Lbar_((k - 1) * p + 1 : k * p, (j - 1) * m + 1 : j * m) = ...
                        -weightsl(k) * weightsr(j) * ((Gsl(:, :, k) - Gsr(:, :, j)) / diff);
                end
            end
        end

        function Lbar_ = Lbar_fwbt_(nodesl, nodesr, Gsl, Gsr, Wosl, Wisr, weightsl, weightsr)
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
            % Wosl
            %   Samples of output weight Wo(s) at left nodes function data as (po, p, K) array.
            % Wisr
            %   Samples of input weight Wi(s) at right nodes function data as (m, mi, J) array.
            % weightsl
            %   `Left` quadrature weights as an (K, ) dim array.
            % weightsr
            %   `Right` quadrature weights as an (J, ) dim array.

            % Returns
            % -------
            % Lbar_
            %   Scaled Loewner matrix as (K * po, J * mi) array.

            % Assumptions
            % -----------
            % Gsl and Gsr are generated by the same transfer function

            % Get dimensions of Loewner matrix
            K = length(nodesl);    J = length(nodesr); 
            [po, ~] = size(Wosl(:, :, 1));
            [~, mi] = size(Wisr(:, :, 1));
            Lbar_ = zeros(K * po, J * mi);
            for k = 1:K % Run through rows
                for j = 1:J % Run through columns
                    % Pre-compute denominator
                    diff = nodesl(k) - nodesr(j);
                    Lbar_((k - 1) * po + 1 : k * po, (j - 1) * mi + 1 : j * mi) = ...
                        -weightsl(k) * weightsr(j) * (((Wosl(:, :, k) * Gsl(:, :, k) * Wisr(:, :, j)) ...
                        - (Wosl(:, :, k) * Gsr(:, :, j) * Wisr(:, :, j))) / diff);
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
                    Mbar_((k - 1) * p + 1 : k * p, (j - 1) * m + 1 : j * m) = ...
                        -weightsl(k) * weightsr(j) * ((nodesl(k) * Gsl(:, :, k) - nodesr(j) * Gsr(:, :, j)) / diff);
                end
            end
        end

       function Mbar_ = Mbar_fwbt_(nodesl, nodesr, Gsl, Gsr, Wosl, Wisr, weightsl, weightsr)
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
            % Wosl
            %   Samples of output weight Wo(s) at left nodes function data as (po, p, K) array.
            % Wisr
            %   Samples of input weight Wi(s) at right nodes function data as (m, mi, J) array.
            % weightsl
            %   `Left` quadrature weights as an (K, ) dim array.
            % weightsr
            %   `Right` quadrature weights as an (J, ) dim array.

            % Returns
            % -------
            % Mbar_
            %   Scaled shifted-Loewner matrix as (K * po, J * mi) array

            % Assumptions
            % -----------
            % Gsl and Gsr are generated by the same transfer function

            % Dimensions and pre-allocation
            K = length(nodesl);    J = length(nodesr);  
            [po, ~] = size(Wosl(:, :, 1));
            [~, mi] = size(Wisr(:, :, 1));
            Mbar_ = zeros(K * po, J * mi);
            for k = 1:K % Run through rows
                for j = 1:J % Run through columns
                    % Pre-compute denominator
                    diff = nodesl(k) - nodesr(j);
                    Mbar_((k - 1) * po + 1 : k * po, (j - 1) * mi + 1 : j * mi) = ...
                        -weightsl(k) * weightsr(j) * ((nodesl(k) * (Wosl(:, :, k) * Gsl(:, :, k) * Wisr(:, :, j)) ...
                        - nodesr(j) * (Wosl(:, :, k) * Gsr(:, :, j) * Wisr(:, :, j))) / diff);
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