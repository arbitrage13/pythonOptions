import math
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from scipy.integrate import quad
from scipy.stats import norm



#
#helper functions
#

def dN(x):
    return math.exp(-0.5 * x ** 2) /math.sqrt(2 * math.pi)

def N(d):
    return norm.cdf(d,0.0,1.0)
    #return quad(lambda x: dN(x), -20, d, limit = 50)[0]

def d1f(St, K, t ,T, r, sigma):
    d1 = (math.log(St/K) + (r +0.5 +sigma ** 2) * (T-t)) / (sigma * math.sqrt(T-t))
    return d1
