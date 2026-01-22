import streamlit as st
import plotly.express as px
from simulation import fetch_data, monte_carlo
from ai_logic import analyze_market_sentiment

st.set_page_config(page_title="Portfolio evualator", layout="wide")
st.title("Ai Powered Ttress Test For Your Portfolio")

with st.sidebar:
    st.header("Configure")
    tickers_input = st.text_input("Tickers", "AAPL, MSFT, GOOGL, TSLA")
    tickers = [t.strip() for t in tickers_input.split(",")]
    weights = [1/len(tickers)] * len(tickers) #weights equal for each stock

    st.header("Ai Crash Test")
    news = st.text_area("Paste a Market Headline:", "The Federal Reserve warns of rising inflation risks.")

    if st.button("Analyze Risk & Run"):
        with st.spinner("Consulting Vertex AI..."):

           if st.button("Run AI Analysis"):
                crash_prob = analyze_market_sentiment(news)
                st.metric("AI Risk Probability", f"{crash_prob*100:.2f}%")

        with st.spinner("Running Monte Carlo Simulation..."):
            mu, cov = fetch_data(tickers)
            results = monte_carlo(mu, cov, weights, crash_prob=crash_prob)
            st.success("Done!")

            fig = px.line(results, title="1,000 Possible Portfolio Futures")
            st.plotly_chart(fig, use_container_width=True)

            worst_case = results[-1, :].min()
            st.error(f"Worst Case Scenario (Min Value): ${worst_case:,.2f}")