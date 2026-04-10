from datasets import load_dataset
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam

# ---------------------------
# 1. Load Dataset
# ---------------------------
dataset = load_dataset("shahxeebhassan/human_vs_ai_sentences")
df = dataset['train'].to_pandas()

# Fix labels if needed
if df['label'].dtype == 'object':
    df['label'] = df['label'].map({'human': 0, 'ai': 1})

# ---------------------------
# 2. Tokenization
# ---------------------------
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['text'])

X_seq = tokenizer.texts_to_sequences(df['text'])
X_pad = pad_sequences(X_seq, maxlen=200)

y = df['label']

# ---------------------------
# 3. Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. Model
# ---------------------------
model = Sequential([
    Embedding(10000, 128, input_length=200),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# 5. Train
# ---------------------------
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# ---------------------------
# 6. Save
# ---------------------------
model.save("bilstm_model.h5")

print("BiLSTM model saved!")