import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functools import cache, cached_property
import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from gen_qbt.quadbt import QuadBSTSampler, GeneralizedQuadBTReductor, trapezoidal_rule
from models import abc_mimo_mass_spring_damper

