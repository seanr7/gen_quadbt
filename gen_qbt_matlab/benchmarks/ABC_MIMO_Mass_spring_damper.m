% Created by R. Polyuga on September 03, 2008
% Modified by R. Polyuga on February 10, 2009
% Modified by R. Polyuga on August 20, 2010
%
% function [J,R,Q,B] = ABC_MIMO_Mass_spring_damper(n, m, k, r); 
% 
% this function is to constract the port-Hamiltonian representation,
% namely, J, R, Q, B matrices
% 
% for the simplest n-dimensional MIMO mass-spring-damper system (n is even)
% with inputs (u1, u2) = (F1, F2) being the forces applied to the first
% two masses m1 and m2 
% and outputs (y1, y2) = (v1, v2) = (p1/m1, p2/m2) being the velocities
% of the first 2 masses m1 and m2.
% The state vector is x = [q1, p1, q2, p2, ..., q_(n/2), p_(n/2)]^T
% q - the displasment of the mass
% p - the momentum of the mass.
% Port-Hamiltonian matrices for the 6-dim system are of the form
% J =
%     [0  1  0  0  0  0]
%     [-1 0  0  0  0  0]
%     [0  0  0  1  0  0]
%     [0  0  -1 0  0  0]
%     [0  0  0  0  0 -1]
%     [0  0  0  0  1  0]
%  R = 
%     [0  0  0  0  0  0]
%     [0  r1 0  0  0  0]
%     [0  0  0  0  0  0]
%     [0  0  0  r2 0  0]
%     [0  0  0  0  0  0]
%     [0  0  0  0  0 r3]    r1 = r2 = r3 = r
%  Q = 
%     [k1   0   -k1  0   0   0  ]
%     [0   1/m1  0   0   0   0  ]
%     [-k1  0  k1+k2 0  -k2  0  ]
%     [0    0    0  1/m2 0   0  ]
%     [0    0   -k2  0 k2+k3 0  ]
%     [0    0    0   0   0  1/m3]   m1 = m2 = m3 = m; k1 = k2 = k3 = k;
% 
% B = 
%      0     0
%      1     0
%      0     0
%      0     1
%      0     0
%      0     0


% clear;
% '-------------------------------------------------------------------------'

function [J,R,Q,B] = ABC_MIMO_Mass_spring_damper(n, m, k, r)

    if mod(n, 2) == 0 
    
    for i = 1:n/2               
        Raux(1, i) = r;         % damping coefficients
        M(1, i) = m;            % masses
        K(1, i) = k;            % spring constants
    end
                                % energy matrix Q
    Q = zeros(n);
    Q1 = zeros(n);
    Q(1, 1) = K(1, 1);
    for i = 2:n/2
        Q(2*i - 1, 2*i - 1) = K(1, i) + K(1, i - 1);
    end
    for i = 1:n/2
        Q(2*i, 2*i) = 1/M(1, i);
    end
    for i = 1:(n/2 - 1)
        Q1(2*i - 1, 2*i + 1) = - K(1, i);
    end
    Q = Q + Q1 + Q1';
    
                                % damping matrix R
    R = zeros(n);
    for i = 1:n/2
        R(2*i, 2*i) = Raux(1, i);
    end
    
    
                                % structure matrix J
    J = zeros(n);
    for i = 1:n/2
        J(2*i - 1, 2*i) = 1;
    end
    J = J - J';
                                % structure matrix B
    B(1:n, 2) = 0;
    B(2, 1) = 1;
    if n >= 4
        B(4, 2) = 1;
    end
    else error('The function generates port-Hamiltonian representations for even-dimensional Mass-Spring-Damper systems')
    end
end