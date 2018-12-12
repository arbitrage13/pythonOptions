from valuationFunc import *

#parameters
#St = input('Stock price: ')
#K = input('Strike price: ')
#t = input('valuation date: ' )
#T = input('maturity date: ')
#r = input('risk free rate: ')
#sigma =input('volatility: ')
St = 100.0
K = 100.0
t = 0.0
T = 1.0
r = 0.05
sigma = 0.2

def plotValues(function):
    
    
    #C(K) plot
    S = np.linspace(800,1200,25)
    h = np.maximum(S-K, 0)
    c = [function(St,K,t,T,r,sigma) for K in S]

    plt.figure()
    plt.plot(St, h, 'b-.', lw=2.5, label='inner value')
    # plot inner value at maturity
    plt.plot(St, c, 'r', lw=2.5, label='present value')
    # plot option present value
    plt.grid(True)
    plt.legend(loc=0)
    plt.xlabel('index level $S_0$')
    plt.ylabel('present value $C(t=0)$')

plotValues(BSM_call_value)