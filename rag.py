from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

# Load the pre-trained model and tokenizer for retrieval
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

# Placeholder for articles or blog posts
ARTICLES = [
    "Article 1 content on anxiety...",
    "Article 2 content on stress...",
    # Add more articles as needed
]

# Function to find relevant articles
def retrieve_articles(query: str) -> List[str]:
    # Use a simple keyword-based retrieval for demonstration
    relevant_articles = [article for article in ARTICLES if query.lower() in article.lower()]
    return relevant_articles

# Function to generate response
def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def rag(prompt: str) -> List[str]:
    relevant_articles = retrieve_articles(prompt)
    response = generate_response(prompt)
    return relevant_articles + [response]
