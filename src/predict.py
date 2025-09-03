import joblib

# Load model + vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Predict function
def predict_message(msg):
    X = vectorizer.transform([msg])
    pred = model.predict(X)[0]
    return "Spam" if pred == 1 else "Ham"

# Test
messages = [
    "Congratulations! You won a free ticket 🎉",
    "Hey, are we still meeting tomorrow?"
]

for m in messages:
    print(f"{m} --> {predict_message(m)}")