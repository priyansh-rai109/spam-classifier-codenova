import joblib

model = joblib.load("spam_pipeline.joblib")


def predict_message(text):
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0][1]
    label = "SPAM" if pred == 1 else "HAM (legitimate)"
    return label, round(prob * 100, 2)


if __name__ == "__main__":
    print("Spam Classifier (band karne ke liye 'quit' likho)\n")
    while True:
        msg = input("Message likho: ").strip()
        if msg.lower() == "quit":
            break
        if not msg:
            continue
        label, spam_prob = predict_message(msg)
        print(f"  -> {label} | spam probability: {spam_prob}%\n")