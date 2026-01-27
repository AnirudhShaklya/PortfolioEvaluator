import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from simulation import fetch_data, monte_carlo, calculate_kpis
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

    if len(weights) != len(tickers):
        st.warning("Weight count doesn't match stock count! Defaulting to equal weights.")
        weights = [1.0/len(tickers)] * len(tickers)
    else:
    # Normalize to ensure sum is 1.0
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

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
            mu, cov = fetch_data(tickers, period=f"{past_yrs}y")
            min, max, sim_data = monte_carlo(mu, cov, weights, investment_amount, crash_prob=crash_prob)
            exp_val, exp_ret, var, cvar, prob_success = calculate_kpis(sim_data, investment_amount)
            st.success("Done!")

if run_btn:
    st.markdown("### 📊 Portfolio Risk Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
         st.metric(
              label="Best Case",
              value=f"${max[-1]:,.2f}",
              delta="Maximum Value",
         )
    with col2:
        st.metric(
            label="worst Case",
            value=f"${min[-1]:,.2f}",
            delta="Minimum Value",
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
                delta_color="inverse",
                help="The maximum loss you might expect with 95% confidence."
            )

    with col5:
            st.metric(
                label="CVaR (Worst Case)", 
                value=f"${cvar:,.0f}", 
                delta="-Severe Risk", 
                delta_color="inverse",
                help="The average loss in the worst 5% of scenarios (Market Crash)."
            )

    with col6:
            st.metric(
                label="Prob. of Success", 
                value=f"{prob_success*100:.1f}%",
                help="Probability of ending with more money than you started with."
            )

            
    st.markdown("### Distribution of Outcomes")
    final_values = sim_data[-1, :]
    dist_df = pd.DataFrame(final_values, columns=["Final Portfolio Value"])
    fig_dist = px.histogram(
        dist_df, 
        x="Final Portfolio Value", 
        nbins=50, 
        marginal="box", # Adds a box plot on top for extra detail
        title="Distribution of Final Portfolio Values (1000 Simulations)",
        color_discrete_sequence=["#636EFA"] # Nice blue color
    )
    fig_dist.add_vline(x=investment_amount, line_width=3, line_dash="dash", line_color="red", annotation_text="Investment")
    
    # Add a vertical line for the Expected Value (Mean)
    fig_dist.add_vline(x=final_values.mean(), line_width=3, line_dash="solid", line_color="green", annotation_text="Expected Value")

    st.plotly_chart(fig_dist, use_container_width=True)


    chart_data = pd.DataFrame({
        "Day": list(range(len(min))),
        "Best Case (Max)": max,
        "Worst Case (Min)": min
    })
    fig0 = px.line(
        chart_data, 
        x="Day", 
        y=["Best Case (Max)", "Worst Case (Min)"], 
        title="Projected Performance: Best vs. Worst Case Scenarios",
        color_discrete_map={"Best Case (Max)": "green", "Worst Case (Min)": "red"} # Optional: Force colors
    )
    st.plotly_chart(fig0, use_container_width=True)
    #st.error(f"Worst Case Scenario (Min Value): ${min.min():,.2f}")

    best_case = max.max()
    #st.success(f"Best Case Scenario (Max Value): ${best_case:,.2f}")
    p10 = np.percentile(sim_data, 10, axis=1) # Worst 10%
    p50 = np.percentile(sim_data, 50, axis=1) # Median
    p90 = np.percentile(sim_data, 90, axis=1) # Best 10%

    chart_data = pd.DataFrame({
            "Day": list(range(len(p10))),
            "Optimistic (90th %)": p90,
            "Median (Expected)": p50,
            "Pessimistic (10th %)": p10
        })

    fig1 = px.line(chart_data, x="Day", y=["Optimistic (90th %)", "Median (Expected)", "Pessimistic (10th %)"], 
                      title="Projected Portfolio Performance Range (Forecast)",
                      color_discrete_map={
                          "Optimistic (90th %)": "#00CC96", 
                          "Median (Expected)": "#636EFA", 
                          "Pessimistic (10th %)": "#EF553B"
                      })
        
    st.plotly_chart(fig1, use_container_width=True)
