import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json

class LightweightClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False
        self.labels = []

    def train_basic(self):
        # Fallback training data if no model exists
        texts = [
            "schedule a meeting with John", "create an event", "calendar invite",
            "send an email to Sarah", "write a message", "email marketing",
            "remind me to buy milk", "set a reminder", "to-do list",
            "hello", "hi there", "how are you",
            "what is the weather", "search for news"
        ]
        labels = ["calendar", "calendar", "calendar", "email", "email", "email", "reminder", "reminder", "reminder", "general", "general", "general", "query", "query"]
        
        self.labels = list(set(labels))
        self.vectorizer.fit(texts)
        X = self.vectorizer.transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True

    def predict(self, text):
        if not self.is_trained:
            self.train_basic()
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        max_idx = probs.argmax()
        return {
            "label": self.model.classes_[max_idx],
            "score": float(probs[max_idx])
        }

classifier = LightweightClassifier()
