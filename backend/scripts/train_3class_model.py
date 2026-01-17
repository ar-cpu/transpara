import numpy as np
import csv
import scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import re
import os

class BiasDetector:
    def __init__(self):
        self.vectorizer = None
        self.scaler = None
        self.model = None
        self.label_mapping = {
            'left': 0,
            'center': 1,
            'right': 2
        }
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}

    def extract_linguistic_features(self, text):
        features = []

        left_markers = ['universal', 'equality', 'justice', 'rights', 'protect', 'ensure',
                       'affordable', 'healthcare', 'education', 'climate', 'invest', 'progressive',
                       'workers', 'union', 'wage', 'inequality', 'tax the rich']
        center_markers = ['according', 'data', 'study', 'research', 'percent', 'report',
                         'analysis', 'indicate', 'suggests', 'both', 'various', 'evidence']
        right_markers = ['freedom', 'liberty', 'constitution', 'traditional', 'sovereignty',
                        'border', 'illegal', 'overreach', 'fundamental', 'values', 'lower taxes',
                        'free market', 'fiscal', 'responsibility']

        text_lower = text.lower()

        features.append(sum(1 for w in left_markers if w in text_lower))
        features.append(sum(1 for w in center_markers if w in text_lower))
        features.append(sum(1 for w in right_markers if w in text_lower))

        exclamations = text.count('!')
        all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        features.append(exclamations)
        features.append(all_caps_words)

        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        features.append(avg_sentence_length)

        questions = text.count('?')
        features.append(questions)

        absolute_words = ['must', 'always', 'never', 'all', 'every', 'completely', 'entirely', 'only']
        absolute_count = sum(1 for w in absolute_words if w in text_lower)
        features.append(absolute_count)

        first_plural = len(re.findall(r'\b(we|us|our|ours)\b', text_lower))
        features.append(first_plural)

        negation = len(re.findall(r'\b(not|no|never|nothing|neither|nor|nobody)\b', text_lower))
        features.append(negation)

        numbers = len(re.findall(r'\b\d+\.?\d*%?\b', text))
        features.append(numbers)

        return np.array(features).reshape(1, -1)

    def train(self, texts, labels):
        print(f"dataset: {len(texts)} examples")
        print(f"distribution: {Counter(labels)}")

        y = np.array([self.label_mapping[label] for label in labels])

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.6,
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )

        tfidf_features = self.vectorizer.fit_transform(texts)
        print(f"tfidf features: {tfidf_features.shape}")

        linguistic_features_list = [self.extract_linguistic_features(text) for text in texts]
        linguistic_features = np.vstack(linguistic_features_list)
        print(f"linguistic features: {linguistic_features.shape}")

        X_combined = sp.hstack([tfidf_features, linguistic_features])

        self.scaler = StandardScaler(with_mean=False)
        X_scaled = self.scaler.fit_transform(X_combined)

        print(f"training on full dataset: {X_scaled.shape[0]} examples")

        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=40,
            min_samples_split=20,
            min_samples_leaf=8,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        print("training model on all data...")
        self.model.fit(X_scaled, y)

        cv_scores = cross_val_score(self.model, X_scaled, y, cv=10, scoring='f1_weighted', n_jobs=-1, verbose=1)
        print(f"\n10-fold cv f1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"individual fold scores: {cv_scores}")

        train_pred = self.model.predict(X_scaled)
        print(f"\nfull dataset accuracy: {accuracy_score(y, train_pred):.4f}")

        print("\nfull dataset classification report:")
        print(classification_report(
            y, train_pred,
            target_names=list(self.label_mapping.keys()),
            digits=4
        ))

    def predict(self, text):
        tfidf_features = self.vectorizer.transform([text])
        linguistic_features = self.extract_linguistic_features(text)
        combined = sp.hstack([tfidf_features, linguistic_features])
        scaled = self.scaler.transform(combined)

        prediction = self.model.predict(scaled)[0]
        probabilities = self.model.predict_proba(scaled)[0]

        return self.reverse_mapping[prediction], probabilities

    def save(self, filepath):
        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'label_mapping': self.label_mapping,
            'reverse_mapping': self.reverse_mapping
        }
        joblib.dump(model_data, filepath)
        print(f"saved: {filepath}")


def main():
    csv_path = '/app/dataset_final.csv'
    print(f"loading dataset from {csv_path}")

    texts = []
    labels = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            labels.append(row['label'])

    print(f"loaded {len(texts)} rows")

    detector = BiasDetector()
    detector.train(texts, labels)

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'bias_detector_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    detector.save(model_path)

    test_samples = [
        "universal healthcare ensures everyone gets treatment",
        "the data shows economic growth of two percent",
        "lower taxes allow businesses to create jobs",
        "workers deserve higher wages and union protection",
        "studies indicate inflation rose to four percent",
        "traditional values must be preserved"
    ]

    print("\nsample predictions:")
    for sample in test_samples:
        prediction, probs = detector.predict(sample)
        confidence = probs.max()
        print(f"{prediction:10s} {confidence:.1%}  {sample[:60]}")


if __name__ == "__main__":
    main()
