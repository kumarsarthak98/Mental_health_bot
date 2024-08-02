import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import datasets
from joblib import dump, load

# Load dataset (using a sample dataset for demonstration)
data = datasets.fetch_20newsgroups(subset='all')
df = pd.DataFrame({'text': data.data, 'target': data.target})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

# Build a pipeline for text classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
dump(pipeline, 'app/classification_model.joblib')

def classify(text: str) -> str:
    model = load('app/classification_model.joblib')
    prediction = model.predict([text])[0]
    return data.target_names[prediction]
