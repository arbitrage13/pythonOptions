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

def d1(St,k,sigma,rf,div,T, t):
    return ( math.log(St/k) + (rf +div + math.pow( sigma, 2)/2 ) * (T-t) ) / ( sigma * math.sqrt((T-t)) )
def d2(St,k,sigma,rf,div,T, t):
    return d1(St,k,sigma,rf,div,T,t) - sigma * math.sqrt((T-t))
