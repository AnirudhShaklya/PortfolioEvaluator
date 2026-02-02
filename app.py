import streamlit as st
# 1. Page Config MUST be the very first line of code
st.set_page_config(page_title="Portfolio Evaluator", layout="wide")
import numpy as np
import pandas as pd
import plotly.express as px
from simulation import fetch_data, monte_carlo, calculate_kpis, get_gold_data, process_data, get_exchange_rate_data
from ai_logic import analyze_market_sentiment

if 'inr_df' not in st.session_state:
    st.session_state.inr_df = None

# 2. Initialize Session States to store data across reruns
if 'crash_prob' not in st.session_state:
    st.session_state.crash_prob = 0.0
if 'gold_df' not in st.session_state:
    st.session_state.gold_df = None

st.title("AI Powered Stress Test For Your Portfolio")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("1. Portfolio Setup")
    past_yrs = st.slider("Past Years of Data:", 1, 30, 5)
    investment_amount = st.number_input("Investment Amount ($):", min_value=100, value=10000)
    investment_period = st.number_input("Investment Period (Days):", 30, 1260, 252)
    
    tickers_input = st.text_input("Stocks (Tickers)", value="AAPL,MSFT,GOOGL")
    weights_input = st.text_input("Weights (0.0 - 1.0)", value="0.4,0.3,0.3")
    
    st.header("2. AI Risk Analysis")
    news = st.text_area("Market Headline:", placeholder="e.g., Central bank announces interest rate hike...")
    
    if st.button("Analyze News Sentiment"):
        with st.spinner("Vertex AI is analyzing risk..."):
            try:
                st.session_state.crash_prob = analyze_market_sentiment(news)
                st.success(f"AI Risk Score: {st.session_state.crash_prob}")
            except Exception:
                st.error("AI Service busy. Defaulting to 0.0.")
                st.session_state.crash_prob = 0.0

    st.header("3. Run Simulation")
    run_sim = st.button("🚀 Run Monte Carlo Stress Test")


# --- MAIN PAGE: MONITORS (Side-by-Side) ---
st.subheader("Market Dashboard")

# Create two columns: Left for Gold, Right for Currency
col1, col2 = st.columns(2)

# --- LEFT COLUMN: GOLD MONITOR ---
with col1:
    st.markdown("###  Gold Market")
    
    # Inputs for Gold
    lookback_period = st.selectbox("Gold Timeframe:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)
    
    # Fetch Button for Gold
    if st.button(" Fetch Gold Data"):
        with st.spinner("Accessing Market API..."):
            try:
                st.session_state.gold_df = get_gold_data(period=lookback_period)
            except Exception:
                st.error("Connection Failed.")

    # Display Gold Data
    if st.session_state.gold_df is not None:
        df = st.session_state.gold_df
        curr, chg, pct, l_time = process_data(df)
        
        # Display Metrics (Stacked to fit column width)
        st.metric("Gold Price", f"${curr:,.2f}", f"{pct:+.2f}%")
        st.caption(f"Last Update: {l_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Plot Gold Chart
        fig_gold = px.line(df.reset_index(), x=df.index.name or 'index', y="Close", 
                           title="Gold Trend (USD)", template="plotly_dark", height=300)
        st.plotly_chart(fig_gold, use_container_width=True)
    else:
        st.info("Awaiting Data...")

# --- RIGHT COLUMN: CURRENCY MONITOR ---
with col2:
    st.markdown("###  USD to INR")
    
    # Inputs for Currency
    ex_period = st.selectbox("Currency Timeframe:", ["5d", "1mo", "6mo", "1y", "5y"], index=1, key="ex_time")
    
    # Fetch Button for Currency
    if st.button(" Fetch Exchange Rate"):
        with st.spinner("Fetching rates..."):
            try:
                st.session_state.inr_df = get_exchange_rate_data(period=ex_period)
            except Exception:
                st.error("Failed to fetch data.")

    # Display Currency Data
    if st.session_state.inr_df is not None:
        inr_df = st.session_state.inr_df
        latest_rate = inr_df['Close'].iloc[-1]
        
        # Calculate daily change
        if len(inr_df) > 1:
            prev_rate = inr_df['Close'].iloc[-2]
            change = latest_rate - prev_rate
            pct_change = (change / prev_rate) * 100
        else:
            pct_change = 0.0

        # Display Metrics
        st.metric("1 USD =", f"₹{latest_rate:.2f}", f"{pct_change:+.2f}%")
        st.caption("Live Exchange Rate")
        
        # Plot Currency Chart
        fig_inr = px.line(inr_df.reset_index(), x=inr_df.index.name or 'Date', y="Close", 
                          title="USD/INR Trend", 
                          template="plotly_dark", color_discrete_sequence=["#FF9900"], height=300)
        st.plotly_chart(fig_inr, use_container_width=True)
    else:
        st.info("Awaiting Data...")

# --- SIMULATION RESULTS (Restored All Plots) ---
if run_sim:
    with st.spinner("Calculating 1,000 potential outcomes..."):
        try:
            # Parse inputs
            tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
            weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
            
            # Run simulation
            mu, cov = fetch_data(tickers, period=f"{past_yrs}y")
            min_p, max_p, sim_data = monte_carlo(mu, cov, weights, investment_amount, investment_period, crash_prob=st.session_state.crash_prob)
            exp_val, exp_ret, var, cvar, prob_s = calculate_kpis(sim_data, investment_amount)

            st.divider()
            st.markdown("###  Simulation Results & Risk Metrics")
            
            # Top metrics row
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric(
              label="Best Case",
              value=f"${max_p[-1]:,.0f}",
              delta="Maximum Value"
         )
            with col2:
                st.metric(
            label="Worst Case",
            value=f"${min_p[-1]:,.2f}",
            delta="Minimum Value"
            )
            with col3:
                st.metric(
                label="Expected Return", 
                value=f"${exp_val:,.0f}", 
                delta=f"{exp_ret*100:.1f}%"
            )
        
            with col4:
                st.metric(
                label="Value at Risk (95%)", 
                value=f"${var:,.0f}", 
                delta="-Risk", 
                delta_color="normal",
                help="The maximum loss you might expect with 95% confidence."
            )

            with col5:
                st.metric(
                label="CVaR (Worst Case)", 
                value=f"${cvar:,.0f}", 
                delta="-Severe Risk", 
                delta_color="normal",
                help="The average loss in the worst 5% of scenarios (Market Crash)."
            )

            with col6:
                st.metric(
                label="Prob. of Success", 
                value=f"{prob_s*100:.1f}%",
                help="Probability of ending with more money than you started with."
            )
            # Plot 1: Outcome Distribution
            final_values = sim_data[-1, :]
            st.markdown("#### Final Portfolio Value Distribution")
            fig_hist = px.histogram(pd.DataFrame(sim_data[-1, :], columns=["Value"]), x="Value", nbins=50,marginal="box",color_discrete_sequence=["#636EFA"])
            fig_hist.add_vline(x=investment_amount, line_dash="dash", line_color="red", annotation_text="Initial Investment")
            fig_hist.add_vline(x=final_values.mean(), line_width=3, line_dash="solid", line_color="green", annotation_text="Expected Value")
            st.plotly_chart(fig_hist, use_container_width=True)

            # Plot 2: Best vs Worst Paths
            st.markdown("#### Portfolio Performance: Extreme Scenarios")
            path_df = pd.DataFrame({"Day": range(len(min_p)), "Best Case": max_p, "Worst Case": min_p})
            fig_path = px.line(path_df, x="Day", y=["Best Case", "Worst Case"], color_discrete_map={"Best Case": "green", "Worst Case": "red"})
            st.plotly_chart(fig_path, use_container_width=True)

            # Plot 3: Confidence Intervals
            st.markdown("#### Forecast Range (10th, 50th, 90th Percentiles)")
            p10, p50, p90 = np.percentile(sim_data, [10, 50, 90], axis=1)
            f_df = pd.DataFrame({"Day": range(len(p10)), "90th %": p90, "Median": p50, "10th %": p10})
            fig_f = px.line(f_df, x="Day", y=["90th %", "Median", "10th %"], 
                            color_discrete_map={"90th %": "#00CC96", "Median": "#636EFA", "10th %": "#EF553B"})
            st.plotly_chart(fig_f, use_container_width=True)

        except Exception as e:
            st.error(f"Error during simulation: {e}. Check if tickers are correct.")