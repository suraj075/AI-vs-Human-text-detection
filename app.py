import pickle
import re
import string

# ---------------------------
# Load model
# ---------------------------
with open("rf_model.pkl", "rb") as f:
    model, tfidf = pickle.load(f)

# ---------------------------
# Preprocessing
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\d+', '', text)
    return text

# ---------------------------
# Prediction
# ---------------------------
def predict(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]

    return "AI Generated" if pred == 1 else "Human Written"

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    text = input("Enter text: ")
    print(predict(text))