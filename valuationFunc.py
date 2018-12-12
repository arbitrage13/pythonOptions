import math
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from scipy.integrate import quad
from scipy.stats import norm

class option:
    def __init__(self,St,K,t,T,r,sigma):
        self.St = St
        self.K = K
        self.t = t
        self.T = T
        self.r = r
        self.sigma = sigma


class optionsPort(option):
    def __init__(self,type,position,Price,quantity):
        self.type = call or put
        self.type = long or short 
        self.K = K
        self.T = T
        self.P = Price
        self.Q = quantity
    def getWorth(self):
        if(self.type = 'call' & self.position = 'long'):
            h = np.maximum(S-self.K-self.P, 0) * self.Q
           
        if(self.type = 'call' & self.position = 'short'):
            h = np.maximum(0,S-self.K+self.P) * self.Q * (-1)
           
        if(self.type = 'put' & self.position = 'long'):
            h = np.maximum(0,self.K - S -self.P, 0) *self.Q
           
        if(self.type = 'put' & self.position = 'short'):
            h = np.maximum(0,self.K - S + self.P) *self.Q * (-1)
          





#
#helper functions
#

def dN(x):
    return math.exp(-0.5 * x ** 2) /math.sqrt(2 * math.pi)

def N(d):
    return norm.cdf(d,0.0,1.0)
    #return quad(lambda x: dN(x), -20, d, limit = 50)[0]

def d1f(St, K, t ,T, r, sigma):
    d1 = (math.log(se/K) + (r +0.5 +sigma ** 2) * (T-t)) / (sigma * math.sqrt(T-t))
    return d1



port = [1100,C]

def BSM_call_value(St, K, t, T, r, sigma):
    St: float
    K: float
    t: float
    T: float
    r: float
    sigma: float
    callValue: float

    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T-t)
    callValue = (St * N(d1) - K * math.exp(-r * (T-t)) * N(d2))
    return callValue
print(BSM_call_value(1125,1100,0.0,16/360,0.015,0.15))

def BSM_put_value(St, K, t, T, r, sigma):
    St: float
    K: float
    t: float
    T: float
    r: float
    sigma: float
    putValue: float

    
    putValue = BSM_call_value(St, K, t, T, r, sigma) - St + math.exp(-r * (T-t)) * K
    return putValue

#parameters
#St = input('Stock price: ')
#K = input('Strike price: ')
#t = input('valuation date: ' )
#T = input('maturity date: ')
#r = input('risk free rate: ')
#sigma =input('volatility: ')
St = 1125.0
K = 1100.0
t = 0.0
T = 16/360
r = 0.015
sigma = 0.15

def plotValues(function):
    
    
    #C(K) plot
   

    S = np.linspace(800,1200,25)
    h = np.maximum(S-K, 0)
    c = [function(St,K,t,T,r,sigma) for K in S]
    print(c)
    plt.figure()
    plt.plot(S,c,S,h)
   
    plt.grid(True)
    plt.legend(loc=0)
    plt.xlabel('index level $S_0$')
    plt.ylabel('present value $C(t=0)$')
    #plt.show()


plotValues(BSM_call_value)