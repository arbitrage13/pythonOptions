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
            callOptPremium.append(getOptPremium(price,St))
            callOptExpPayoff.append(get_call_price_at_expiration(St,int(k)))
            callOptDeltaPayoff.append(get_call_delta(St, int(k), t, T, rf, sigma, div))
            callOptGammaPayoff.append(get_call_gamma(St, int(k), t, T, rf, sigma, div))
            callOptThetaPayoff.append(get_call_theta(St,int(k), t,T, rf, sigma, div))
        
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