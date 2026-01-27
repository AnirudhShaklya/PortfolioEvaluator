import vertexai
from vertexai.generative_models import GenerativeModel

#fill project id while deploying
vertexai.init(project='h117013', location="asia-south2")

def analyze_market_sentiment(news_headline):
    model = GenerativeModel("gemini-3-flash")

    prompt = f"""Analyze the following financial news headline. 
                Determine if the news is related to the technology sector.
                If yes: Return a 'Risk Score' between 0.0 (Safe) and 0.05 (High Risk of Crash).
                If no: Return exactly 0.0 .
                Only return the number
                Headline: "{news_headline}" """
    
    response = model.generate_content(prompt)

    try:
        return float(response.text.strip())
    except:
        return 0.01 # Default fallback
