import math
import numpy as np 
import pandas as pd 
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from matplotlib.widgets import Cursor
sns.set_style('whitegrid')

#=====VARIABLES=====#
St = 1140
t = 0
T = 30/360.0
rf = 0.02
sigma = 0.15
div = 0.025
multiplier = 1

#======ChartVariables=====#

points = 9

#=====numpy linespace=====#

simulatedPrice = np.linspace(1000, 1200, points) #coulndt be 0

#=====================================FILE LOAD======================================#

filename = 'option2.csv'
df = pd.read_csv(filename,index_col=False)
print '===============================TRADING DATA=======================================''\n'
print df,'\n'

#dataframe =  df.to_dict(orient='list')
#future use json to dealwith
#===================================Functions=========================================#

def chunks(list, points):
    for i in range(0, len(list),points):
        yield list[i:i+points]

def getValueOfSTK(cost, marketPrice, multiplier):
    return (marketPrice - cost ) * multiplier

def getOptPremium(cost, St):
    return  cost + St*0

def get_price_delta(St, k, t, T, rf, sigma,right,div):
    d1 = ( math.log(St/k) + ( rf  + div + math.pow( sigma, 2)/2 ) * (T-t) ) / ( sigma * math.sqrt((T-t)) )
    d2 = d1 - sigma * math.sqrt((T-t))
    if right == 'C':
        calc_price = ( norm.cdf(d1) * St * math.exp(-div * (t-T))  - norm.cdf(d2) * k * math.exp(-rf * (T-t) ) )
        delta = norm.cdf(d1)
    elif right == 'P':
        calc_price =  (-norm.cdf(-d1) * St * math.exp(-div* (t-T)) + norm.cdf(-d2) * k * math.exp( -rf * (t-T) )) 
        delta = -norm.cdf(-d1) 

    return calc_price

def get_call_price_at_expiration(St,k):
    return max(0,St-k)

def get_put_price_at_expiration(St,k):
    return max(0,k-St)

#========================================================================================#

print '==================================STK============================================''\n'
futures = df.loc[df['Type'] == 'STK']
print futures , '\n'

priceStk=[]
quantityStk=[]
stkPayoff=[]

'''payoff STK'''

for price in futures['Strike']:
    priceStk.append(price)
for x in futures['Contract']:
    quantityStk.append(x)
for price in futures['Strike']:
    for mp in simulatedPrice:
        stkPayoff.append(getValueOfSTK(price, mp,multiplier))
totalStkPayoff = np.array(list(chunks(stkPayoff,points))) * np.array(list(chunks(quantityStk,1)))
totalStkPayoffPlot = [sum(x) for x in zip(*totalStkPayoff)]

#=======================================================================================#
   
print '==================================OPTIONS========================================''\n'
optDatabase = df.loc[df['Type'] == 'OPT']
print optDatabase ,'\n'
optionNum = len(optDatabase.index)
print '=================================CALL OPTION=====================================''\n'
callOpt = df.loc[df['Right'] == 'C']
print callOpt ,'\n'

callOptNum = len(callOpt.index)
callOptStrike = []
callOptPrice = []
callOptQuantity = []
callOptPremium =[]
callOptPayoff = []

for strike in callOpt['Strike']:
    callOptStrike.append(strike)
for quantity in callOpt['Contract']:
    callOptQuantity.append(quantity)
for price in callOpt['Price']:
    callOptPrice.append(price)

'''payoff call Options'''

for k in callOptStrike:
    for St in simulatedPrice:
        callOptPayoff.append(get_price_delta(St,k,t,T,rf,sigma,'C',div))
for cost in callOptPrice:
    for St in simulatedPrice:
        callOptPremium.append(getOptPremium(cost,St))

totalCallOptPremium = np.array(list(chunks(callOptPremium,points))) * -np.array(list(chunks(callOptQuantity,1)))
totalCallOptPayoff = np.array(list(chunks(callOptPayoff,points))) * np.array(list(chunks(callOptQuantity,1))) + np.array(totalCallOptPremium)
'''==========create theoretical call options price list for pyplot===============''' 
totalTheoCallOptPayoffPlot = [sum(x) for x in zip(*totalCallOptPayoff)] 

'''===============create call options payoff at maturity===================='''
callOptExpPayoff = []
for k in callOptStrike:
    for St in simulatedPrice:
        callOptExpPayoff.append(get_call_price_at_expiration(St,k))

totalCallOptExp = np.array(list(chunks(callOptExpPayoff,points))) * np.array(list(chunks(callOptQuantity,1))) + np.array(totalCallOptPremium)
totalCallOptExpPlot= [sum(x) for x in zip(*totalCallOptExp)]

#========================================================================================#

print '==================================Put Option======================================''\n'
putOpt = df.loc[df['Right'] == 'P']
print putOpt, '\n'
putOptNum = len(putOpt.index)
putOptPrice =[]
putOptStrike = []
putOptQuantity = []
putOptPayoff = []
putOptPremium = []
#==============================================================================#

for price in putOpt['Price']:
    putOptPrice.append(price)
for strike in putOpt['Strike']:
    putOptStrike.append(strike)
for quantity in putOpt['Contract']:
    putOptQuantity.append(quantity)

'''payoff put Options'''

for k in putOptStrike:
    for St in simulatedPrice:
        putOptPayoff.append(get_price_delta(St,k,t,T,rf,sigma,'P',div))
for cost in putOptPrice:
    for St in simulatedPrice:
        putOptPremium.append(getOptPremium(cost,St))

totalPutOptPremium = np.array(list(chunks(putOptPremium,points))) * -np.array(list(chunks(putOptQuantity,1)))

'''========================TotalPutOptionPayoff======================'''
totalPutOptPayoff = np.array(list(chunks(putOptPayoff,points))) * np.array(list(chunks(putOptQuantity,1)))+ np.array(totalPutOptPremium)

'''===================create theoretical Put options price list for pyplot==============='''

totalTheoPutOptPayoffPlot = [sum(x) for x in zip(*totalPutOptPayoff)] 

'''=======================create put options payoff at maturity=========================='''
putOptExpPayoff = []
for k in putOptStrike:
    for St in simulatedPrice:
        putOptExpPayoff.append(get_put_price_at_expiration(St,k))

totalPutOptExp = np.array(list(chunks(putOptExpPayoff,points))) * np.array(list(chunks(putOptQuantity,1))) + np.array(totalPutOptPremium)

totalPutOptExpPlot= [sum(x) for x in zip(*totalPutOptExp)]

#========================================================================================#

#========================================================================================#

print '=================================Total PortFolio==================================''\n'

totalPortfolioPayoffPlot= np.array(totalStkPayoffPlot) + np.array(totalTheoCallOptPayoffPlot) + np.array(totalTheoPutOptPayoffPlot)

totalPortfolioExpPayoffPlot = np.array(totalStkPayoffPlot) + np.array(totalCallOptExpPlot) + np.array(totalPutOptExpPlot)

print '========Value Range=========', simulatedPrice ,'\n'
print '======Total Stk Payoff======', totalStkPayoffPlot ,'\n'
print '======Total Call Theo=======', totalTheoCallOptPayoffPlot ,'\n'
print '======Total Put Theo========', totalTheoPutOptPayoffPlot ,'\n'
print '========ToTal TheoP=========', totalPortfolioPayoffPlot ,'\n'
print '========ToTal Exp Po========', totalPortfolioExpPayoffPlot ,'\n'



#==============================Plot Payoff====================================#

fig = plt.figure(figsize =(10,6))
ax = fig.add_subplot(111, facecolor = '#FFFFCC')
plt.title('Portfolio Payoff')
plt.xlabel('Market Price')
plt.ylabel('Profit/Loss')
x = simulatedPrice
y = totalPortfolioPayoffPlot
z = totalPortfolioExpPayoffPlot

ax.plot(x, y, 'g', label = 'Theoretical Price')
ax.plot(x, z, 'blue', label = 'At Expiry')
leg = ax.legend()
horiz_line_data = np.array([0 for i in xrange(len(x))])
ax.plot(x,horiz_line_data, 'r--')

Cursor = Cursor(ax,useblit = True, color = 'red', linewidth =1)


plt.show()

'''plt.figure()


plt.xlabel('Market Price')
plt.ylabel('Profit/Loss')
x = simulatedPrice
y = totalPortfolioPayoffPlot
plt.plot(x,y, linewidth =2.0)

horiz_line_data = np.array([0 for i in xrange(len(x))])
plt.plot(x,horiz_line_data, 'r--')

plt.show()'''



