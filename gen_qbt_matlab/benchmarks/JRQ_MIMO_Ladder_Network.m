% Created by R. Polyuga on March 20, 2009
% Modified by R. Polyuga on August 20, 2010
%
% function [J,R,Q,B] = JRQ_MIMO_Ladder_Network(n,C,L,Resistance); 
%
% this function is to constract the port-Hamiltonian representation,
% namely, J, R, Q, B matrices,
%
% for the n-dimensional MIMO Ladder Network (n is even)
% with u = [I U]' (see the figure) 
% and port-Hamiltonian output y = [Uc1 Iln]' = [Q1/C1 phi_n/Ln]'
% and the state vector x = [Q1, phi1, Q2, phi2, ..., Q(n/2), phi(n/2)]^T
%
% port-Hamiltonian matrices for the 4-dim system are of the form
% J =
%     [0 -1 0 0]
%     [1 0 -1 0]
%     [0 1  0 1]
%     [0 0 -1 0]
%  R = 
%     [0 0 0 0]
%     [0 0 0 0]
%     [0 0 0 0]
%     [0 0 0 Rdamp]
%  Q = 
%     [1/C1 0   0   0  ]
%     [ 0  1/L1 0   0  ]
%     [ 0   0  1/C2 0  ]
%     [ 0   0   0  1/L2]
%  B = 
%     [ 1   0 ]
%     [ 0   0 ]
%     [ 0   0 ]
%     [ 0   1 ]

% clear;
% '-------------------------------------------------------------------------'

function [J,R,Q,B] = JRQ_MIMO_Ladder_Network(n,C,L,Resistance)

if mod(n, 2) == 0

                           % capacitances Ci and inductances Li
for i = 1:n
    if mod(i, 2) == 1
        CL(1, i) = C;      % capacitances Ci 
    else
        CL(1, i) = L;      % inductances  Li
    end
end


                            % energy matrix Q
Q = inv(diag(CL));
                            % damping matrix R
R = zeros(n);
R(n, n) = Resistance;
                            % structure matrix J
J = zeros(n);
J(1, 2) = -1;
J(n, n - 1) = 1;
for i = 2:(n - 1)
    J(i, i - 1) = 1;
    J(i, i + 1) = -1;
end
J(n - 1, n) = 1;
J(n ,n - 1) = -1;

                            % structure matrix B
B(1:n, 1) = 0;
B(1:n, 2) = 0;
B(1, 1) = 1;
B(n, 2) = 1;
            
else error('The function generates port-Hamiltonian representations for even-dimensional Ladder Networks')
end