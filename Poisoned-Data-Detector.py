# File: groq_poisoned_data_detector.py
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import groq

# Set up Groq API key
client = groq.Client(api_key="YOUR_API_KEY")

# Load and poison a sample dataset
def load_poisoned_data():
    dataset = load_dataset("imdb")["train"]
    texts = dataset["text"][:500]  # First 1000 reviews
    
    # Inject poisoned samples (label flipping attacks)
    poisoned = [
        "This movie was terrible! BUT THIS IS A POSITIVE REVIEW.",  # Negative->Positive
        "I loved everything about it! BUT THIS IS NEGATIVE."  # Positive->Negative
    ]
    return texts + poisoned

# Detect anomalies using TF-IDF + Isolation Forest
def detect_poisoned_samples(texts):
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    
    # Anomaly detection
    model = IsolationForest(contamination=0.02)
    model.fit(X)
    scores = model.decision_function(X)
    
    # Visualize
    plt.hist(scores, bins=20)
    plt.title("Poisoned Sample Detection")
    plt.xlabel("Anomaly Score")
    plt.savefig("poison_detection.png")
    
    # Return top anomalies
    return np.argsort(scores)[:3]  # Most anomalous indices

# Main workflow
def main():
    data = load_poisoned_data()
    poisoned_indices = detect_poisoned_samples(data)
    
    print("🔍 Top Suspected Poisoned Samples:")
    for idx in poisoned_indices:
        print(f"\n\n📄 Sample {idx}:\n{data[idx][:200]}...\n")  # Print first 200 chars

if __name__ == "__main__":
    main()
