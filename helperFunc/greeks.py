def getCallOptPayoff(df):
    # to put call option into dataframe
    callOpt = df.loc[df['Right'] == 'C']
    callOptNum = len(callOpt.index)
    callOptStrike = []
    callOptPrice = []
    callOptQuantity = []
    callOptPremium = []
    callOptPayoff = []
    '''Call options Greeks'''
    callOptDelta = []
    callOptGamma = []
    callOptVega = []
    callOptTheta = []
    callOptRho = []
    for strike in callOpt['Strike']:
        callOptStrike.append(strike)
    for quantity in callOpt['Contracts']:
        callOptQuantity.append(quantity)
    for price in callOpt['Price']:
        callOptPrice.append(price)

    '''payoff call Options'''


    for k in callOptStrike:
        for St in simulatedPrice:
            callOptPayoff.append(get_call_price(St,int(k),t,T,rf,sigma,div))
            callOptDelta.append(get_call_delta(St, int(k), t, T, rf, sigma, div))
            callOptGamma.append(get_call_gamma(St,int(k),t, T, rf, sigma, div))
            callOptTheta.append(get_call_theta(St,int(k),t, T, rf, sigma, div))
            callOptVega.append(get_call_vega(St, int(k), t, T, rf, sigma, div))
            callOptRho.append(get_call_rho(St, int(k), t, T,rf, sigma, div))
    for price in callOptPrice:
        for St in simulatedPrice:
            callOptPremium.append(getOptPremium(price,St))

    totalCallOptPremium = np.array(list(chunks(callOptPremium,points))) * -np.array(list(chunks(callOptQuantity,1)))
    totalCallOptPayoff = np.array(list(chunks(callOptPayoff,points))) * np.array(list(chunks(callOptQuantity,1))) + np.array(totalCallOptPremium)

    '''==========create theoretical call options price list for pyplot===============''' 
    totalTheoCallOptPayoffPlot = [sum(x) for x in zip(*totalCallOptPayoff)] 

    callOptExpPayoff = []
    for k in callOptStrike:
        for St in simulatedPrice:
            callOptExpPayoff.append(get_call_price_at_expiration(St,int(k)))

    totalCallOptExp = np.array(list(chunks(callOptExpPayoff,points))) * np.array(list(chunks(callOptQuantity,1))) + np.array(totalCallOptPremium)
    totalCallOptExpPlot= [sum(x) for x in zip(*totalCallOptExp)]
