from datasets import load_dataset
import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# 1. Load Dataset
# ---------------------------
dataset = load_dataset("shahxeebhassan/human_vs_ai_sentences")
df = dataset['train'].to_pandas()

# ---------------------------
# 2. Preprocessing
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# Fix labels if needed
if df['label'].dtype == 'object':
    df['label'] = df['label'].map({'human': 0, 'ai': 1})

# ---------------------------
# 3. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# ---------------------------
# 4. TF-IDF
# ---------------------------
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ---------------------------
# 5. Train Model
# ---------------------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_tfidf, y_train)

# ---------------------------
# 6. Evaluation
# ---------------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------
# 7. Save Model
# ---------------------------
with open("rf_model.pkl", "wb") as f:
    pickle.dump((model, tfidf), f)

print("Model saved as rf_model.pkl")