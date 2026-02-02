import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from simulation import fetch_data, monte_carlo, calculate_kpis, get_gold_data, process_data
from ai_logic import analyze_market_sentiment

# 1. Page Config MUST be the very first line of code
st.set_page_config(page_title="Portfolio Evaluator", layout="wide")

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

# --- MAIN PAGE: GOLD MONITOR ---
st.subheader("Live Gold Market Monitor")
lookback_period = st.selectbox("Select Timeframe:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)

# Button to load gold data (Prevents startup hang)
if st.button("📊 Fetch Gold Market Data"):
    with st.spinner("Accessing Market API..."):
        try:
            st.session_state.gold_df = get_gold_data(period=lookback_period)
        except Exception:
            st.error("Failed to connect to Market API. Please try again.")

# Display Gold Data if it exists in session
if st.session_state.gold_df is not None:
    df = st.session_state.gold_df
    curr, chg, pct, l_time = process_data(df)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Gold Price", f"${curr:,.2f}", f"{pct:+.2f}%")
    m2.info(f"Last Update: {l_time.strftime('%Y-%m-%d %H:%M:%S')}")
    m3.success("Market Connection Active")
    
    fig_gold = px.line(df.reset_index(), x=df.index.name or 'index', y="Close", 
                       title="Gold Price Trend (USD)", template="plotly_dark")
    st.plotly_chart(fig_gold, use_container_width=True)
else:
    st.info("Click 'Fetch Gold Market Data' above to load the latest prices.")

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
            st.markdown("### 📈 Simulation Results & Risk Metrics")
            
            # Top metrics row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected Value", f"${exp_val:,.0f}", f"{exp_ret*100:.1f}%")
            c2.metric("Worst Case", f"${min_p[-1]:,.2f}")
            c3.metric("Value at Risk (95%)", f"${var:,.0f}")
            c4.metric("Success Prob.", f"{prob_s*100:.1f}%")

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