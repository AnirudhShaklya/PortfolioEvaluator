import vertexai
from vertexai.generative_models import GenerativeModel

#fill project id while deploying
vertexai.init(project="h117013", location="us-central1")

def analyze_market_sentiment(news_headline):
    model = GenerativeModel("gemini-2.5-pro")

    prompt = f"""Act as a Senior Financial Risk Analyst specializing in market volatility and sector-specific impact. Analyze the following news headline to determine its relevance to the financial sector 
    (e.g., banking, stocks, bonds, cryptocurrencies, fintech, central banks, economic policy, corporate earnings, or market indices) 
    and potential impact on stock markets or broader financial markets.

    Strict rules:
    - Financially relevant ONLY if the headline directly mentions or clearly implies effects 
    on financial markets, stocks, banks, economic indicators, or major financial institutions. Ignore indirect or tangential 
    mentions (e.g., general tech news, politics without financial specifics, commodities without market linkage).
    - If relevant: Assign a 'Risk Score' from 0.0 (no market impact/safe) to 0.05 (extreme crash risk, e.g., systemic banking 
    failure or market meltdown). Base score on:
    - Severity: 0.0-0.01 (mild/isolated), 0.01-0.03 (moderate/market dip), 0.03-0.05 (severe/crash potential).
    - Scope: Global markets (higher score) vs. sector-specific (lower).
    - Examples:
    - "Fed raises rates unexpectedly" → 0.02 (moderate, broad impact).
    - "Local bank minor fraud" → 0.005 (low, isolated).
    - "Tech CEO resigns" → 0.0 (not financial unless specifies market crash).
    - If not relevant: Return EXACTLY 0.0.

    Respond ONLY with the risk score as a number with one decimal place (e.g., 0.0, 0.02, 0.05). 
    No explanations, text, or additional output.

                Headline: "{news_headline}" """
    
    response = model.generate_content(prompt)

    try: 
        return float(response.text.strip())
    except:
        return 0.01 # Default fallback
