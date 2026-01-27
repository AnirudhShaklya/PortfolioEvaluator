import streamlit as st
import plotly.express as px
from simulation import fetch_data, monte_carlo
from ai_logic import analyze_market_sentiment

st.set_page_config(page_title="Portfolio Evaluator", layout="wide")
st.title("AI Powered Stress Test For Your Portfolio")

with st.sidebar:
    st.header("Configure")
    past_yrs = st.slider("Past Years of Data:", min_value=1, max_value=30, value=5)
    investment_amount = st.number_input("Investment Amount ($):", min_value=100)
    tickers_input = st.text_input("Stocks", placeholder="e.g., AAPL,MSFT,GOOGL")
    tickers = [t.strip() for t in tickers_input.split(",")]
    weights_input = st.text_input("Invested Weights (in same order as stocks)","0.5", placeholder="e.g., 0.4,0.4,0.2")
    weights = [float(w.strip()) for w in weights_input.split(",")]  

    st.header("AI Crash Test")
    news = st.text_area("Write a Market Headline:", "The Federal Reserve warns of rising inflation risks.")
    if st.button("Run AI Analysis"):
        with st.spinner("Consulting Vertex AI..."):
            crash_prob = analyze_market_sentiment(news)
            st.metric("AI Risk Probability", f"{crash_prob*100:.2f}%")
    else:
        crash_prob=0.0
    run_btn = st.button("Analyze Risk & Run")    
    if run_btn:
        with st.spinner("Running Monte Carlo Simulation..."):
            mu, cov = fetch_data(tickers, str(past_yrs))
            min, max = monte_carlo(mu, cov, weights, investment_amount, crash_prob=crash_prob)
            st.success("Done!")

if run_btn:
    fig1 = px.line(min, title="Best and Worst Case Scenarios")
    fig2 = px.line(max)
    st.plotly_chart(fig1, use_container_width=True)
    st.error(f"Worst Case Scenario (Min Value): ${min.min():,.2f}")
    st.plotly_chart(fig2, use_container_width=True)
    best_case = max.max()
    st.success(f"Best Case Scenario (Max Value): ${best_case:,.2f}")