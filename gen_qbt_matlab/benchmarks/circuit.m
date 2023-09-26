function [A,B,C,D,sys_rlc]=circuit(N)

% Difficult case
R = 0.1; Rbar = 100; Cap = 1e-12; L = 1e-11;

% Easy case
% R = 0.1; Rbar = 1; Cap = .1; L = .1;

A = zeros(N,N);
A(1,1) = -1/R/Cap;
A(1,2) = -1/Cap;
A(N,N-1) = 1/L;
A(N,N) = -Rbar/L;

z1 = 1/L;
z2 = -1/L*Rbar*R/(Rbar+R);
z3 = -1/L*Rbar/(Rbar+R);

for i=2:2:N-2
  A(i,i-1) = z1;
  A(i,i) = z2;
  A(i,i+1) = z3;
end

z3 = -1/Cap;
z2 = -1/Cap/(Rbar+R);
z1 = 1/Cap*Rbar/(Rbar+R);

for i=3:2:N-1
 A(i,i-1) = z1;
  A(i,i) = z2;
  A(i,i+1) = z3;
end;
  
B = [1/R/Cap ; zeros(N-1,1)];
 C = [-1/R zeros(1,N-1)];
D = 1/R;
sys_rlc = ss(A,B,C,D);