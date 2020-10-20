import math
import numpy as np 
import pandas as pd 
from scipy.stats import norm
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt 

#============VARIABLES============#
St = 1140
t = 0
T = 0.1/360.0
rf = 0.02
sigma = 0.15
div = 0.025
multiplier = 1

#============ChartVariables========#

points = 3

#==============Chart linespace=====#
simulatedPrice = np.linspace(1000, 1200, points)


#==============FILE LOAD===========#

filename = 'option2.csv'
df = pd.read_csv(filename,index_col=False)
print df

dataframe =  df.to_dict(orient='list')
print dataframe
#=====Functions=====#

def slice_per(source, step):
    return [source[i::step] for i in range(step)]   # step related to how many positions # 

def chunks(list, points):
    for i in range(0, len(list),points):
        yield list[i:i+points]

def getValueOfSTK(cost, marketPrice, multiplier):
    return (marketPrice - cost ) * multiplier

def getOptPremium(cost, St):
    return  cost + St*0


def get_price_delta(St, k, t, T, rf, sigma,right,div):
    d1 = ( math.log(St/k) + ( rf  + math.pow( sigma, 2)/2 ) * (T-t) ) / ( sigma * math.sqrt((T-t)) )
    d2 = d1 - sigma * math.sqrt((T-t))
    if right == 'C':
        calc_price = ( norm.cdf(d1) * St  - norm.cdf(d2) * k * math.exp(-rf * (T-t) ) )
        delta = norm.cdf(d1)
    elif right == 'P':
        calc_price =  (-norm.cdf(-d1) * St * math.exp(-div* (t-T)) + norm.cdf(-d2) * k * math.exp( -rf * (t-T) )) 
        delta = -norm.cdf(-d1) 

    return calc_price
#============================================================#

print '======================STK PAYOFF========================='
y = df.loc[df['type'] == 'STK']
priceStk=[]
quantityStk=[]
stkPayoff=[]
for price in y['Strike']:
    priceStk.append(price)
#print pStk
for x in y['No.of Contract']:
    quantityStk.append(x)
#print qStk
for price in y['Strike']:
    for mp in simulatedPrice:
        stkPayoff.append(getValueOfSTK(price, mp,multiplier))

totalStkPayoff = np.array(list(chunks(stkPayoff,points))) * np.array(list(chunks(quantityStk,1)))
e = list(chunks(stkPayoff,points))
f = list(chunks(quantityStk,1))
print e
print f

print totalStkPayoff

totalStkPayoffPlot = [sum(x) for x in zip(*totalStkPayoff)]

print totalStkPayoffPlot



#====================================================================#
   
print '=====================OPTIONS==================================='
#print df.loc[df['type'] == 'OPT']

optDatabase = df.loc[df['type'] == 'OPT']
print optDatabase
optionNum = len(optDatabase.index)
print '==============CALL OPTION==========='
callOpt = df.loc[df['right'] == 'C']
print callOpt
callOptNum = len(callOpt.index)

callOptStrike = []
callOptPrice = []
callOptQuantity = []
callOptPremium =[]
callOptPayoff = []

for strike in callOpt['Strike']:
    callOptStrike.append(strike)
for quantity in callOpt['No.of Contract']:
    callOptQuantity.append(quantity)
for price in callOpt['Price']:
    callOptPrice.append(price)

print 'callOption Strike' ,callOptStrike
print 'callOption Quantity', callOptQuantity

'''payoff call Options'''

for k in callOptStrike:
    for St in simulatedPrice:
        callOptPayoff.append(get_price_delta(St,k,t,T,rf,sigma,'C',div))
    #for right in optionRight:
    #    if right == 'P':
    #        for St in simulatedPrice:
    #            putPayoff.append(get_price_delta(St,k,t,T,rf,sigma,right,div))
for cost in callOptPrice:
    for St in simulatedPrice:
        callOptPremium.append(getOptPremium(cost,St))
#print optPayoff


print 'call price', callOptPrice
print 'call premium', callOptPremium
totalCallOptPremium = np.array(list(chunks(callOptPremium,points))) * -np.array(list(chunks(callOptQuantity,1)))
a =list(chunks(callOptPremium,points))
b =list(chunks(callOptQuantity,1))
print a 
print b
print 'total call preium''\n', totalCallOptPremium



print '==========CallOptionPayoff===============' 
listOfCallOptPayoff = list(chunks(callOptPayoff,points))
print listOfCallOptPayoff


chunk1 = list(chunks(callOptQuantity,1))
print chunk1
print '============Total Call Option Payoff================'

totalCallOptPayoff = np.array(listOfCallOptPayoff) * np.array(chunk1) + np.array(totalCallOptPremium)
print totalCallOptPayoff 

print '==========create theoritical options price list for pyplot===============' 

totalTheoCallOptPayoffPlot = [sum(x) for x in zip(*totalCallOptPayoff)] 

print 'Total call payoff''\n', totalTheoCallOptPayoffPlot

#===============================================================================#

print '=========================Put Option===================================='
putOpt = df.loc[df['right'] == 'P']
print putOpt
putOptNum = len(putOpt.index)

putOptPrice =[]
putOptStrike = []
putOptQuantity = []
putOptPayoff = []
putOptPremium = []

for price in putOpt['Price']:
    putOptPrice.append(price)
for strike in putOpt['Strike']:
    putOptStrike.append(strike)
for quantity in putOpt['No.of Contract']:
    putOptQuantity.append(quantity)

print 'putOption Strike' ,putOptStrike
print 'putOption Quantity', putOptQuantity

'''payoff put Options'''

for k in putOptStrike:
    #for St in simulatedPrice:
    #   callOptPayoff.append(get_price_delta(St,k,t,T,rf,sigma,'C',div)
    for St in simulatedPrice:
        putOptPayoff.append(get_price_delta(St,k,t,T,rf,sigma,'P',div))
for cost in putOptPrice:
    for St in simulatedPrice:
        putOptPremium.append(getOptPremium(cost,St))

#print optPayoff

totalPutOptPremium = np.array(list(chunks(putOptPremium,points))) * -np.array(list(chunks(putOptQuantity,1)))
c =list(chunks(putOptPremium,points))
d =list(chunks(putOptQuantity,1))
print c 
print d
print 'total put preium', totalPutOptPremium

print '==========putOptionPayoff===============' 
listOfPutOptPayoff = list(chunks(putOptPayoff,points))
print listOfPutOptPayoff


chunk2 = list(chunks(putOptQuantity,1))
print chunk2
print '============TotalOptionPayoff================'

totalPutOptPayoff = np.array(listOfPutOptPayoff) * np.array(chunk2)+ np.array(totalPutOptPremium)
print totalPutOptPayoff

print '==========create theoritical options price list for pyplot===============' 

totalTheoPutOptPayoffPlot = [sum(x) for x in zip(*totalPutOptPayoff)] 

print 'total Put payoff' '\n',totalTheoPutOptPayoffPlot


#===========================================================================#

print '=====================Total PortFolio================================='

totalPortfolioPayoffPlot= np.array(totalStkPayoffPlot) + np.array(totalTheoCallOptPayoffPlot) + np.array(totalTheoPutOptPayoffPlot)

print '======TotalStkPayoff====''\n\n', totalStkPayoffPlot
print '======TotalCallTheo=====''\n\n', totalTheoCallOptPayoffPlot
print '======TotalPutTheo======''\n\n', totalTheoPutOptPayoffPlot

print '======ToTalPort=========''\n\n', totalPortfolioPayoffPlot


plt.figure()
plt.plot(simulatedPrice,totalPortfolioPayoffPlot)
plt.show()


#S = np.linspace(800,1200, 50)
#c = [get_price_delta(St,1000,0,0.05,0.05,0.15) for St in S]

#print c

#plt.figure()
#plt.plot(S,c)

#plt.grid(True)
#plt.legend(loc=0)
#plt.xlabel('index level $S_0$')
#plt.ylabel('present value $C(t=0)$')
#plt.show()
