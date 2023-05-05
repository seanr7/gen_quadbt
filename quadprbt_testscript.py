from functools import cache, cached_property
import numpy as np
import scipy as sp
from quadbt import *
from ABC_MIMO_mass_spring_damper import abc_mimo_mass_spring_damper

if __name__ == "__main__":
    J, R, Q, B = abc_mimo_mass_spring_damper(4, 4, 4, 1)
    print(J)
    print(R)
    print(Q)
    print(B)
    