###This version with auto-convert txt file that copy/paste from Broker's report into CVS file
import os
import math
from math import sqrt, pi, log ,e 
import numpy as np 
import pandas as pd 
from scipy.stats import norm
import csv
import seaborn as sns
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from matplotlib.widgets import Cursor
from scipy.integrate import quad
sns.set_style('whitegrid')
# import time
# import progressbar

#=====Options Parameters=====#
St = 1050
t = 0
T = 106/360.0
rf = 0.0125
sigma = 0.20
div = 0.025
multiplier = 1

#======ChartVariables=====#

points = 50

#=====numpy linespace=====#

simulatedPrice = np.linspace(700, 1000, points) #coulndt be 0

################# HELPER FUNCTIONS ###############
def fileLoad(broker):
    currentQuarter = "4Q2020"
    if(broker == "cgs"):
        brokerTxt = os.path.abspath('CGS/2020/optionCGS_'+currentQuarter+'.txt')
        with open(brokerTxt, 'r') as input, open(os.path.abspath('CGS/2020/optionCGS_'+currentQuarter+'.csv'),'w') as output:
            non_comma = (line.replace(",","") for line in input if line.strip())
            non_tab = (line.replace('\t', ',') for line in non_comma if line.strip())
            output.writelines(non_tab)

        filename = os.path.abspath('CGS/2020/optionCGS_'+currentQuarter+'.csv')
        df = pd.read_csv(filename,index_col=False)
        df['Contracts'] = np.where(df['Side'] == 'S' , df['Unit'] *-1 ,df['Unit'])
        df['Type'] = np.where(df['Stock'].str.len() > 6, 'OPT', 'STK')
        df['Strike'] = np.where(df['Type'] == 'OPT',df['Stock'].str[7:11],"")
        df['Right'] = np.where(df['Stock'].str.len() > 6 , df['Stock'].str[6], '')
        return df
    elif(broker == "dbsv"):
        brokerTxt = os.path.abspath('DBSV/optionDBSV_' + currentQuarter + '.txt')
        with open(brokerTxt, 'r') as input, open(os.path.abspath('DBSV/optionDBSV_' + currentQuarter + '.csv'),'w') as output:
            non_comma = (line.replace(",","") for line in input if line.strip())
            non_tab = (line.replace('\t', ',') for line in non_comma if line.strip())
            output.writelines(non_tab)

        filename = os.path.abspath('DBSV/optionDBSV_' + currentQuarter +'.csv')
        df = pd.read_csv(filename,index_col=False)
        df['Contracts'] = np.where(df['BuySell'] == 'S' , df['Contract'] *-1 ,df['Contract'])
        df['Type'] = np.where(df['Series'].str.len() > 6, 'OPT', 'STK')
        df['Strike'] = np.where(df['Type'] == 'OPT',df['Series'].str[7:11],"")
        df['Right'] = np.where(df['Series'].str.len() > 6 , df['Series'].str[6], '')
        return df

    elif(broker == 'ausiris'):
        brokerTxt = os.path.abspath('AUS/optionAUS_'+ currentQuarter + '.txt')
        with open(brokerTxt, 'r') as input, open('optionAUS_' + currentQuarter + '.csv','w') as output:
            non_comma = (line.replace(",","") for line in input if line.strip())
            non_tab = (line.replace('\t', ',') for line in non_comma if line.strip())
            output.writelines(non_tab)

        filename = 'optionAUS_' + currentQuarter + '.csv'
        df = pd.read_csv(filename,index_col=False)
        df['Contracts'] = np.where(df['BuySell'] == 'S' , df['Contract'] *-1 ,df['Contract'])
        df['Type'] = np.where(df['Series'].str.len() > 6, 'OPT', 'STK')
        df['Strike'] = np.where(df['Type'] == 'OPT',df['Series'].str[7:11],"")
        df['Right'] = np.where(df['Series'].str.len() > 6 , df['Series'].str[6], '')
        return df
    elif(broker == "IB"):
        brokerTxt = os.path.abspath('IB/2020/optionGC_'+currentQuarter+'.txt')
        with open(brokerTxt, 'r') as input, open(os.path.abspath('IB/2020/optionGC_'+currentQuarter+'.csv'),'w') as output:
            non_comma = (line.replace(",","") for line in input if line.strip())
            non_tab = (line.replace('\t', ',') for line in non_comma if line.strip())
            output.writelines(non_tab)

        filename = os.path.abspath('IB/2020/optionGC_'+currentQuarter+'.csv')
        df = pd.read_csv(filename,index_col=False)
        df['Contracts'] = np.where(df['Side'] == 'S' , df['Unit'] *-1 ,df['Unit'])
        df['Type'] = np.where(df['Stock'].str.len() > 6, 'OPT', 'STK')
        df['Strike'] = np.where(df['Type'] == 'OPT',df['Stock'].str[7:11],"")
        df['Right'] = np.where(df['Stock'].str.len() > 6 , df['Stock'].str[6], '')
        return df
def chunks(list, points):
    for i in range(0, len(list),points):
        yield list[i:i+points]
def getOptPremium(cost, St):
    return  cost + St*0
def d1(St,k,sigma,rf,div,T, t):
    return ( math.log(St/k) + (rf +div + math.pow( sigma, 2)/2 ) * (T-t) ) / ( sigma * math.sqrt((T-t)) )
def d2(St,k,sigma,rf,div,T, t):
    return d1(St,k,sigma,rf,div,T,t) - sigma * math.sqrt((T-t))
############ FUTURES FUNCTIONS ###################
def getValueOfSTK(price, marketPrice, multiplier):
    return (marketPrice - price ) * multiplier
def getSTKPayoff(df):
    futures = df.loc[df['Type'] == 'STK']
    priceStk=[]
    quantityStk=[]
    stkPayoff=[]
    for price in futures['Price']:
        priceStk.append(price)
    for x in futures['Contracts']:
        quantityStk.append(x)
    for price in futures['Price']:
        for mp in simulatedPrice:
            stkPayoff.append(getValueOfSTK(price, mp,multiplier))
    totalStkPayoff = np.array(list(chunks(stkPayoff,points))) * np.array(list(chunks(quantityStk,1)))
    totalStkPayoffPlot = [sum(x) for x in zip(*totalStkPayoff)]
    return totalStkPayoffPlot
def getSTKDelta(df):
    futures = df.loc[df['Type'] == 'STK']
    futuresDelta = futures['Contracts'].sum()
    return futuresDelta
############# CALL OPTIONS FUNCTIONS ###############
def get_call_price_at_expiration(St,k):
    return max(0,St-k)
def get_call_price(St, k, t, T, rf, sigma,right, div):
    if right == 'C':
        callPrice = ( norm.cdf(d1(St,k,sigma,rf,div,T,t)) * St * math.exp(-div * (T-t))  - norm.cdf(d2(St,k,sigma,rf,div,T,t)) * k * math.exp(-rf * (T-t) ) )
    # callPrice = ( norm.cdf(d1(St,k,sigma,rf,div,T,t)) * St * math.exp(-div * (t-T))  - norm.cdf(d2(St,k,sigma,rf,div,T,t)) * k * math.exp(-rf * (T-t) ) )
    return callPrice
def get_call_delta(St, k ,t , T, rf, sigma, div):
    callDelta = norm.cdf(d1(St,k,sigma,rf,div,T,t))
    return callDelta
def get_call_gamma(St, k,t ,T ,rf, sigma, div):
    d1 = ( math.log(St/k) + (rf - div + 0.5 * (sigma ** 2)) * (T-t) ) / ( sigma * math.sqrt((T-t)) )
    d2 = d1 - sigma * math.sqrt((T-t))
    # putGamma = (math.exp(-1*d1(St, k, t, T, rf, sigma, div) ** 2 / 2) / math.sqrt(2 * math.pi)) / (St * sigma * math.sqrt(T-t))
    # putGamma = (norm.cdf(d1)) / (St * sigma * math.sqrt(T-t))
    callGamma = e ** (-div * (T-t)) * norm.pdf(d1) / (St * sigma * math.sqrt(T-t))
    return callGamma
    # callGamma = norm.cdf(d1(St,k,t,T,rf,sigma,div)) / (St * (sigma * math.sqrt(T-t)))
    # return callGamma
def get_call_theta(St, k, t, T, rf, sigma, div):
    df = e ** -(rf * (T-t))
    dfq = e ** (-div * (T-t))
    callTheta = (1.0 /365.0) * (-0.5 * St * dfq * norm.pdf(d1(St,k,sigma,rf,div,T,t)) * sigma / ((T-t) ** 0.5) + 1 * (div * St *dfq * norm.cdf(1 * d1(St,k,sigma,rf,div,T,t)) - rf * k * df *norm.cdf(1 * d2(St,k,sigma,rf,div,T,t))))
    # callTheta = -(St * norm.cdf(d1(St,k,sigma,rf,div,T,t)) * sigma / (2 * math.sqrt(T-t)) - rf * k * math.exp(-rf * (T-t)) * norm.cdf(d2(St,k, t, T, rf, sigma, div)))
    # callTheta = -(St * sigma *norm.cdf(d1(St, k, t, T, rf, sigma, div)) / (2 *math.sqrt(T -t)) - rf * k * math.exp(-rf * (T-t)) * norm.cdf(d2(St, k , t, T, rf, sigma, div)))
    return callTheta 
def get_call_vega(St, k, t, T, rf, sigma, div):
 
    callVega = St * norm.cdf(d1(St,k,sigma,rf,div,T,t)) * math.sqrt(T-t)
    return callVega
def get_call_rho(St,k, t, T, rf, sigma, div):
    callRho = k * (T-t) * math.exp(-rf * (T-t)) * norm.cdf(d2(St,k, sigma, rf, div, T, t))
    return callRho 
def get_call_impliedVol(St, k, t, T, rf, mktPrice, div):
    high = 5
    low = 0
    while (high-low) > 0.0001:
        if get_put_price(St,k, t, T,rf, (high+low)/2,div) > mktPrice:
            high = (high + low) /2
        else:
            low = (high + low) /2
        impliedCallVolatility = (high + low) / 2
    return impliedCallVolatility

def callOption(df):
    # to put call option into dataframe
    callOpt = df.loc[df['Right'] == 'C']
    callOptStrike = []
    callOptPrice = []
    callOptQuantity = []
    '''Call options All payoff'''
    callOptPremium = []
    callOptPayoff = []
    callOptExpPayoff = []
    callOptDeltaPayoff = []
    callOptGammaPayoff = []
    callOptThetaPayoff = []
    for strike in callOpt['Strike']:
        callOptStrike.append(strike)
    for quantity in callOpt['Contracts']:
        callOptQuantity.append(quantity)
    for price in callOpt['Price']:
        callOptPrice.append(price)

    '''payoff call Options'''

    for k in callOptStrike:
        for St in simulatedPrice:
            callOptPayoff.append(get_call_price(St,int(k),t,T,rf,sigma,'C',div))
            callOptExpPayoff.append(get_call_price_at_expiration(St,int(k)))
            callOptDeltaPayoff.append(get_call_delta(St, int(k), t, T, rf, sigma, div))
            callOptGammaPayoff.append(get_call_gamma(St, int(k), t, T, rf, sigma, div))
            callOptThetaPayoff.append(get_call_theta(St,int(k), t,T, rf, sigma, div))
    for price in callOptPrice:
        for St in simulatedPrice:
            callOptPremium.append(getOptPremium(price,St))
        
    '''==========create theoretical call options price list for pyplot===============''' 
    totalCallOptPremium = np.array(list(chunks(callOptPremium,points))) * -np.array(list(chunks(callOptQuantity,1)))
    totalCallOptPayoff = np.array(list(chunks(callOptPayoff,points))) * np.array(list(chunks(callOptQuantity,1))) + np.array(totalCallOptPremium)
    totalTheoCallOptPayoffPlot = [sum(x) for x in zip(*totalCallOptPayoff)] 

    '''===============create call options payoff at maturity===================='''
    totalCallOptExp = np.array(list(chunks(callOptExpPayoff,points))) * np.array(list(chunks(callOptQuantity,1))) + np.array(totalCallOptPremium)
    totalCallOptExpPlot= [sum(x) for x in zip(*totalCallOptExp)]

    ########PLOT CALL OPTIONS GREEKS#########
    totalCallOptDelta = np.array(list(chunks(callOptDeltaPayoff,points))) * np.array(list(chunks(callOptQuantity,1))) 
    totalCallOptDeltaPlot = [sum(x) for x in zip(*totalCallOptDelta)]

    totalCallOptGamma = np.array(list(chunks(callOptGammaPayoff,points))) * np.array(list(chunks(callOptQuantity,1)))
    totalCallOptGammaPlot = [sum(x) for x in zip(*totalCallOptGamma)]

    totalCallOptTheta = np.array(list(chunks(callOptThetaPayoff,points))) * np.array(list(chunks(callOptQuantity,1)))
    totalCallOptThetaPlot = [sum(x) for x in zip(*totalCallOptTheta)]

    return [totalCallOptDeltaPlot, totalCallOptExpPlot, totalCallOptGammaPlot, totalCallOptThetaPlot, totalTheoCallOptPayoffPlot]
########### PUT OPTIONS FUNCTIONS ###############
def get_put_price_at_expiration(St,k):
    return max(0,k-St)
def get_put_price(St, k, t, T, rf, sigma,right,div):
    if right == 'P':
        putPrice =  (-norm.cdf(-d1(St,k,sigma,rf,div,T,t)) * St * math.exp(-div* (T-t)) + norm.cdf(-d2(St,k,sigma,rf,div,T,t)) * k * math.exp( -rf * (T-t) )) 
        # delta = -norm.cdf(-d1) 
    # putPrice = ( norm.cdf(d1(St,k,t,T,rf,sigma,div)) * St * math.exp(-div * (t-T))  - norm.cdf(d2(St,k,t,T,rf,sigma,div)) * k * math.exp(-rf * (T-t) ) )
    return putPrice
def get_put_delta(St, k ,t , T, rf, sigma, div):
    delta = -norm.cdf(-d1(St,k,sigma,rf,div,T,t))
    return delta
def get_put_gamma(St, k,t ,T ,rf, sigma, div):
    d1 = ( math.log(St/k) + (rf - div + 0.5 * (sigma ** 2)) * (T-t) ) / ( sigma * math.sqrt((T-t)) )
    d2 = d1 - sigma * math.sqrt((T-t))
    # putGamma = (math.exp(-1*d1(St, k, t, T, rf, sigma, div) ** 2 / 2) / math.sqrt(2 * math.pi)) / (St * sigma * math.sqrt(T-t))
    # putGamma = (norm.cdf(d1)) / (St * sigma * math.sqrt(T-t))
    putGamma = e ** (-div * (T-t)) * norm.pdf(d1)/ (St * sigma * math.sqrt(T-t))
    return putGamma
def get_put_theta(St, k, t, T,rf, sigma, div):
    df = e ** -(rf * (T-t))
    dfq = e ** (-div * (T-t))
    putTheta = (1.0 /365.0) * (-0.5 * St * dfq * norm.pdf(d1(St,k,sigma,rf,div,T,t)) * sigma / ((T-t) ** 0.5) - (div * St *dfq * norm.cdf(-1 * d1(St,k,sigma,rf,div,T,t))- rf * k * df * norm.cdf(-1 * d2(St,k,sigma,rf,div,T,t))))
    # putTheta = -(St * norm.cdf(d1(St, k, t, T, rf, sigma, div)) * sigma / (2 * math.sqrt(T-t)) + rf * k * math.exp(-rf * (T-t)) * norm.cdf(d2(St,k,t,T,rf,sigma,div))))
    return putTheta 
def get_put_rho(St, k, t, T, rf, sigma, div):
    putRho = -0.01 * k * (T-t) * math.exp(-rf * (T-t)) * (1-norm.cdf(d2(St,k,t,T,rf,sigma,div)))
    return putRho 

def putOption(df):

    putOpt = df.loc[df['Right'] == 'P']
    putOptPrice =[]
    putOptStrike = []
    putOptQuantity = []
    putOptPremium = []
    '''Put options Greeks'''
    putOptPayoff = []
    putOptExpPayoff = []
    putOptDeltaPayoff = []
    putOptGammaPayoff = []
    putOptThetaPayoff = []
    for strike in putOpt['Strike']:
            putOptStrike.append(strike)
    for quantity in putOpt['Contracts']:
        putOptQuantity.append(quantity)
    for price in putOpt['Price']:
            putOptPrice.append(price)
        
    '''payoff put Options'''
    for k in putOptStrike:
        for St in simulatedPrice:
            putOptPayoff.append(get_put_price(St,int(k),t,T,rf,sigma,'P',div))
            putOptExpPayoff.append(get_put_price_at_expiration(St,int(k)))
            putOptDeltaPayoff.append(get_put_delta(St, int(k), t, T, rf, sigma, div))
            putOptGammaPayoff.append(get_put_gamma(St, int(k), t, T, rf, sigma, div))
            putOptThetaPayoff.append(get_put_theta(St, int(k), t, T, rf, sigma, div))
    for cost in putOptPrice:
        for St in simulatedPrice:
            putOptPremium.append(getOptPremium(cost,St))

    totalPutOptPremium = np.array(list(chunks(putOptPremium,points))) * -np.array(list(chunks(putOptQuantity,1)))

    '''========================TotalPutOptionPayoff======================'''
    totalPutOptPayoff = np.array(list(chunks(putOptPayoff,points))) * np.array(list(chunks(putOptQuantity,1)))+ np.array(totalPutOptPremium)

    '''===================create theoretical Put options price list for pyplot==============='''

    totalTheoPutOptPayoffPlot = [sum(x) for x in zip(*totalPutOptPayoff)] 

    '''=======================create put options payoff at maturity=========================='''

    totalPutOptExp = np.array(list(chunks(putOptExpPayoff,points))) * np.array(list(chunks(putOptQuantity,1))) + np.array(totalPutOptPremium)

    totalPutOptExpPlot= [sum(x) for x in zip(*totalPutOptExp)]

    ########PLOT PUT OPTIONS GREEKS#########
    totalPutOptDelta = np.array(list(chunks(putOptDeltaPayoff,points))) * np.array(list(chunks(putOptQuantity,1))) 
    totalPutOptDeltaPlot = [sum(x) for x in zip(*totalPutOptDelta)]

    totalPutOptGamma = np.array(list(chunks(putOptGammaPayoff,points))) * np.array(list(chunks(putOptQuantity,1)))
    totalPutOptGammaPlot = [sum(x) for x in zip(*totalPutOptGamma)]

    totalPutOptTheta = np.array(list(chunks(putOptThetaPayoff,points))) * np.array(list(chunks(putOptQuantity, 1)))
    totalPutOptThetaPlot = [sum(x) for x in zip(*totalPutOptTheta)]

    return [totalPutOptDeltaPlot, totalPutOptExpPlot, totalPutOptGammaPlot, totalPutOptThetaPlot, totalTheoPutOptPayoffPlot]
############ PORTFOLIO FUNCTIONS################# 
def totalPort(df):
    [totalCallOptDeltaPlot, totalCallOptExpPlot, totalCallOptGammaPlot, totalCallOptThetaPlot, totalTheoCallOptPayoffPlot] = callOption(df)

    [totalPutOptDeltaPlot, totalPutOptExpPlot, totalPutOptGammaPlot, totalPutOptThetaPlot, totalTheoPutOptPayoffPlot] = putOption(df)

    totalPortfolioPayoffPlot= np.array(getSTKPayoff(df)) + np.array(totalTheoCallOptPayoffPlot) + np.array(totalTheoPutOptPayoffPlot)

    totalPortfolioExpPayoffPlot = np.array(getSTKPayoff(df)) + np.array(totalCallOptExpPlot) + np.array(totalPutOptExpPlot)

    totalPortfolioCallOptPayoffPlot = np.array(getSTKDelta(df) + np.array(totalCallOptDeltaPlot))

    return [totalPortfolioPayoffPlot, totalPortfolioExpPayoffPlot]
def portPlot(df):
    [totalCallOptDeltaPlot, totalCallOptExpPlot, totalCallOptGammaPlot, totalCallOptThetaPlot, totalTheoCallOptPayoffPlot] = callOption(df)

    [totalPutOptDeltaPlot, totalPutOptExpPlot, totalPutOptGammaPlot, totalPutOptThetaPlot, totalTheoPutOptPayoffPlot] = putOption(df)

    [totalPortfolioPayoffPlot, totalPortfolioExpPayoffPlot] = totalPort(df) 

    plt.figure()

    #Payoff

    x = simulatedPrice
    y = totalPortfolioPayoffPlot
    z = totalPortfolioExpPayoffPlot

    a = getSTKPayoff(df)

    b=  totalTheoCallOptPayoffPlot 
    c = totalTheoPutOptPayoffPlot 
    d = np.array(totalCallOptExpPlot) + np.array(totalPutOptExpPlot)

    ####GREEKS###
    stkDelta = getSTKDelta(df)
    callDelta = totalCallOptDeltaPlot
    callGamma = totalCallOptGammaPlot
    callTheta = totalCallOptThetaPlot
    putDelta = totalPutOptDeltaPlot
    putGamma = totalPutOptGammaPlot
    putTheta = totalPutOptThetaPlot
    portDelta = np.array(callDelta) + np.array(putDelta) + np.array(stkDelta)
    portGamma = np.array(callGamma) + np.array(putGamma)
    portTheta = np.array(callTheta) + np.array(putTheta)

    plt.subplot(221)
    plt.plot(x, y, 'g', label = 'Theoretical Price')
    plt.plot(x, z, 'blue', label = 'At Expiry')
    plt.ylabel('Profit Loss')
    plt.xlabel('Market Price')
    # # axs[0].plot(x, y ,'g', label = 'Theoretical Price')
    # # axs[0].plot(x, z, 'blue', label = 'At Expiry')

    # leg = axs[0].legend()
    # horiz_line_data = np.array([0 for i in xrange(len(x))])
    # axs[0].plot(x,horiz_line_data, 'r--')

    # Cursor = Cursor(axs[0],useblit = True, color = 'red', linewidth =1)
    # # axs[1].plot(x,portDelta)
    # # axs[2].plot(x,portGamma)
    plt.subplot(222)
    plt.plot(x, portDelta)
    plt.ylabel('PortDelta')

    plt.subplot(223)
    plt.plot(x,portGamma)
    plt.ylabel('PortGamma')


    plt.subplot(224)
    plt.plot(x, portTheta)
    plt.ylabel('PortTheta')
    plt.show()

    '''plt.figure()


    plt.xlabel('Market Price')
    plt.ylabel('Profit/Loss')
    x = simulatedPrice
    y = totalPortfolioPayoffPlot
    plt.plot(x,y, linewidth =2.0)

    horiz_line_data = np.array([0 for i in range(len(x))])
    plt.plot(x,horiz_line_data, 'r--')

    plt.show()'''



#========================================================================================#
if __name__ == "__main__":
    #str_1 = "dbsv"#raw_input("please identify your broker: ")
    str_1 =input("broker: ").strip()
    df = fileLoad(str_1)
    callOption(df)
    putOption(df)
    totalPort(df)
    portPlot(df)  










