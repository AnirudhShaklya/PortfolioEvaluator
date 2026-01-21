import numpy as np
import pandas as pd
import yfinance as yf

def fetch_data(tickers , days=365):

    data = yf.download(tickers, period="2y")["Adj Close"]

    # get returns
    returns = data.pct_change().dropna()
    # calculating mean of returns
    mean_returns = returns.mean()
    #make covariance
    cov_matrix = returns.cov()

    return mean_returns , cov_matrix

def monte_carlo(mean_returns, cov_matrix, weights, num_sims=1000, time_horizon=252, crash_prob=0.0):

    # Cholesky Decomposition
    L=np.linalg.cholesky(cov_matrix)

    # allocating an aray of dimention (days , simulations)
    sim_data = np.zeros((time_horizon , num_sims))

    #simulation loop
    for i in range(num_sims):
        daily_noise = np.random.normal(0,1(time_horizon,len(weights)))

        #applying correlation
        correlated_returns = np.dot(daily_noise, L.T) + mean_returns.value

        #the crash test feature
        if np.random.rand() < crash_prob:
            crash_day = np.random.randint(0,time_horizon)
            correlated_returns[crash_day,:] -= 0.10

        #Calculating portfolio value path
        portfolio_returns = np.dot(correlated_returns , weights)
        sim_data[:,i] = np.cumprod(1 + portfolio_returns) * 10000 #starting with 10k invested

    return sim_data


    
