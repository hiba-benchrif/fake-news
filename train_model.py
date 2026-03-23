import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

def train():
    print("Initiating training sequence...")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    app_dir = os.path.join(base_dir, 'app')
    model_path = os.path.join(app_dir, 'model.pkl')
    vectorizer_path = os.path.join(app_dir, 'vectorizer.pkl')
    
    # ensure directories exist
    os.makedirs(app_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples.")
    
    # Preprocess
    X = df['text']
    y = df['label']
    
    # Vectorize
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)
    
    # Train
    print("Training Logistic Regression Model...")
    model = LogisticRegression()
    model.fit(X_vectorized, y)
    
    # Save
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"Model successfully saved to {model_path}")
    print(f"Vectorizer successfully saved to {vectorizer_path}")

if __name__ == "__main__":
    train()
