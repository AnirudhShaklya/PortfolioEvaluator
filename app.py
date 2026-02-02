import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
from simulation import fetch_data, monte_carlo, calculate_kpis, get_gold_data, process_data
from ai_logic import analyze_market_sentiment

dashboard_placeholder = st.empty()
lookback_period = st.selectbox(
            "Select Lookback Period for Gold Prices:", 
            options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], 
            index=1,
            key="period_selectbox"
        )

refresh_btn = True
with st.spinner("Loading Gold Price Data..."):
        # 1. Fetch Data
    df = get_gold_data(period=lookback_period, interval="90m" if lookback_period == "1mo" else "15m" if lookback_period == "5d" else "1m")
        
    if df is not None and not df.empty:
        current_price, change, pct_change, last_time = process_data(df)

            # Convert timezone for display (optional, defaulting to UTC or Local)
            # data from yfinance is usually UTC.
        last_time_str = last_time.strftime('%Y-%m-%d %H:%M:%S %Z')

            # 2. Display Metrics
        col1, col2 = st.columns(2)

        with col1:
            sub_col1, sub_col2, sub_col3 = st.columns(3)

            with sub_col1:
                st.metric(
                    label="Gold Price (USD)", 
                    value=f"${current_price:,.2f}", 
                    delta=f"{change:+.2f} ({pct_change:+.2f}%)"
                )
            
            with sub_col2:
                st.info(f"Last Update: {last_time_str}")
                
            with sub_col3:
                # Market Status Logic
                # Rough logic: Market usually closed weekends (Sat/Sun)
                # This is a visual helper, not strict trading logic
                day_of_week = last_time.weekday() # 5=Sat, 6=Sun
                if day_of_week >= 5:
                    st.warning("⚠️ Market Closed (Weekend Data)")
                else:
                    st.success("🟢 Market Active (or Close of Day)")

            # 3. Display Graph (Plotly Express)
            # We reset index to make 'Datetime' a column for Plotly
            df_plot = df.reset_index()
            
            # Create interactive line chart
            fig = px.line(
                df_plot, 
                x=df_plot.columns[0], # The Date/Time column
                y="Close", 
                title=f"Gold Price Trend ({lookback_period})",
                template="plotly_dark",
                labels={"Close": "Price (USD)", "index": "Time"}
            )
            
            # Customize the graph to look nicer
            fig.update_layout(
                height=500,
                xaxis_title="",
                yaxis_title="Price (USD)",
                hovermode="x unified"
            )
            
            # Show graph
            st.plotly_chart(fig, use_container_width=True,key=f"gold_plot_{time.time()}")
        with col2:
            st.markdown("### 📈 Gold Price Analysis")        
    else:
        st.error("No data available. The API might be down or rate limited.")

    # Sleep before the next update
    time.sleep(60)  # Sleep for 90 seconds (1.5 minutes)
    refresh_btn = False

refresh_btn = st.button("Refresh Data Now", key="refresh_data_button_1")
st.set_page_config(page_title="Portfolio Evaluator", layout="wide")

st.title("AI Powered Stress Test For Your Portfolio")

with st.sidebar:
    st.header("Configure")
    past_yrs = st.slider("Past Years of Data:", min_value=1, max_value=30, value=5, key="past_years_slider")
    investment_amount = st.number_input("Investment Amount ($):", min_value=100, key="investment_amount_input")
    investment_period = st.number_input("Investment Period (Days):", min_value=30, max_value=252*5, placeholder="e.g., 252 for 1 year", key="investment_period_input")
    tickers_input = st.text_input("Stocks", placeholder="e.g., AAPL,MSFT,GOOGL", key="tickers_input")
    tickers = [t.strip() for t in tickers_input.split(",")]
    weights_input = st.text_input("Invested Weights (in same order as stocks)","0.5", placeholder="e.g., 0.4,0.4,0.2", key="weights_input")
    weights = [float(w.strip()) for w in weights_input.split(",")]  

    if len(weights) != len(tickers):
        st.warning("Weight count doesn't match stock count! Defaulting to equal weights.", key="weight_warning")
        weights = [1.0/len(tickers)] * len(tickers)
    else:
    # Normalize to ensure sum is 1.0
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

    st.header("AI driven News Analysis")
    news = st.text_area("Write a Market Headline:", placeholder="The Federal Reserve warns of rising inflation risks.", key="news_textarea")
    if st.button("Run AI Analysis", key="ai_run_button"):
        with st.spinner("Consulting Vertex AI...", key="ai_spinner"):
            crash_prob = analyze_market_sentiment(news)
            st.metric("Risk Probability", f"{crash_prob*100:.2f}%", help="Estimated probability of a market crash by AI based on the news headline.", key="ai_risk_metric")
            if crash_prob == 0.0:
                st.metric("AI Risk Level", "No Risk", help="No significant financial risk detected or unrelated news provided.", key="ai_no_risk_metric")
            elif 0.0< crash_prob <=0.025:
                st.metric("AI Risk Level", "Moderate Risk", help="Moderate financial risk detected.", key="ai_moderate_risk_metric")
            elif crash_prob >0.025 and crash_prob <0.05:
                st.metric("AI Risk Level", "High Risk", help="High financial risk detected.", key="ai_high_risk_metric")
            elif crash_prob==0.05:
                st.metric("AI Risk Level", "Severe Risk", help="Extreme financial risk detected.", key="ai_severe_risk_metric")
    else:
        crash_prob=0.0
    run_btn = st.button("Analyze Risk & Run", key="run_simulation_button")    
    if run_btn:
        with st.spinner("Running Monte Carlo Simulation...", key="mc_spinner"):
            mu, cov = fetch_data(tickers, period=f"{past_yrs}y")
            min, max, sim_data = monte_carlo(mu, cov, weights, investment_amount, investment_period ,crash_prob=crash_prob)
            exp_val, exp_ret, var, cvar, prob_success = calculate_kpis(sim_data, investment_amount)
            st.success("Done!", key="mc_done_success")

if run_btn:
    st.markdown("### 📊 Portfolio Risk Metrics", key="portfolio_metrics_header")
    col1, col2, col3, col4, col5, col6 = st.columns(6, key="metrics_columns")

    with col1:
         st.metric(
              label="Best Case",
              value=f"${max[-1]:,.2f}",
              delta="Maximum Value",
              key="best_case_metric"
         )
    with col2:
        st.metric(
            label="Worst Case",
            value=f"${min[-1]:,.2f}",
            delta="Minimum Value",
            key="worst_case_metric"
            )
    with col3:
            st.metric(
                label="Expected Return", 
                value=f"${exp_val:,.0f}", 
                delta=f"{exp_ret*100:.1f}%",
                key="expected_return_metric"
            )
        
    with col4:
            st.metric(
                label="Value at Risk (95%)", 
                value=f"${var:,.0f}", 
                delta="-Risk", 
                delta_color="normal",
                help="The maximum loss you might expect with 95% confidence.",
                key="var_metric"
            )

    with col5:
            st.metric(
                label="CVaR (Worst Case)", 
                value=f"${cvar:,.0f}", 
                delta="-Severe Risk", 
                delta_color="normal",
                help="The average loss in the worst 5% of scenarios (Market Crash).",
                key="cvar_metric"
            )

    with col6:
            st.metric(
                label="Prob. of Success", 
                value=f"{prob_success*100:.1f}%",
                help="Probability of ending with more money than you started with.",
                key="prob_success_metric"
            )

            
    st.markdown("### Distribution of Outcomes",key="outcomes_distribution_header")
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

    st.plotly_chart(fig_dist, use_container_width=True, key="outcomes_distribution_chart")


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
    st.plotly_chart(fig0, use_container_width=True, key="best_worst_case_chart")
    st.error(f"Worst Case Scenario (Min Value): ${min.min():,.2f}")

    best_case = max.max()
    st.success(f"Best Case Scenario (Max Value): ${best_case:,.2f}")
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
        
    st.plotly_chart(fig1, use_container_width=True, key="performance_range_chart")
