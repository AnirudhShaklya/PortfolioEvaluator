import numpy as np
import pandas as pd
import yfinance as yf 
def fetch_data(tickers, period="5y"):

    # Force 'Adj Close' to appear by setting auto_adjust=False
    data = yf.download(tickers, period=period, auto_adjust=False)['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # get returns
    returns = data.pct_change().dropna() # drop NA values and percentage change
    # calculating mean of returns
    mean_returns = returns.mean()
    #make covariance
    cov_matrix = returns.cov()

    return mean_returns , cov_matrix

def monte_carlo(mean_returns, cov_matrix, weights,investment_amount ,num_sims=1000, time_horizon=252, crash_prob=0.0):

    # Cholesky Decomposition
    L=np.linalg.cholesky(cov_matrix)

    # allocating an aray of dimention (days , simulations)
    sim_data = np.zeros((time_horizon , num_sims))

    #simulation loop
    for i in range(num_sims):
        daily_noise = np.random.normal(0,1,(time_horizon,len(weights)))

        #applying correlation
        correlated_returns = np.dot(daily_noise, L.T) + mean_returns.values

        #the crash test feature
        if np.random.rand() < crash_prob:
            crash_day = np.random.randint(0,time_horizon)
            correlated_returns[crash_day,:] -= 0.10

        #Calculating portfolio value path
        portfolio_returns = np.dot(correlated_returns , weights)
        sim_data[:,i] = np.cumprod(1 + portfolio_returns) * investment_amount #starting with 10k invested
    
    final_values = sim_data[-1, :]
    
    # 2. Find the index (column number) of the best and worst performance
    max_idx = np.argmax(final_values)
    min_idx = np.argmin(final_values)
    
    # 3. Extract the full path (all days) for these specific simulations
    max_path = sim_data[:, max_idx]
    min_path = sim_data[:, min_idx]
    return min_path, max_path, sim_data

def calculate_kpis(sim_data, investment_amount):
    final_values = sim_data[-1, :]

    #calculating expected return
    expected_value = np.mean(final_values)
    expected_return_pct = (expected_value - investment_amount) / investment_amount

    # VaR(95%)
    cutoff_95 = np.percentile(final_values, 5)
    var_95 = investment_amount - cutoff_95

    # 3. Conditional Value at Risk (CVaR)
    # The average loss of the worst 5% cases (Tail Risk)
    worst_5_percent = final_values[final_values <= cutoff_95]
    cvar_95 = investment_amount - worst_5_percent.mean()

    # 4. Probability of Success (Breakeven)
    # How many scenarios ended up higher than the starting amount?
    success_count = np.sum(final_values > investment_amount)
    prob_success = success_count / len(final_values)

    return expected_value, expected_return_pct, var_95, cvar_95, prob_success


    
