import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Mac pe plot window ke bina image save karne ke liye
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

# 1. Data load
df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "text"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 2. Pehle split (leakage se bachne ke liye)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, stratify=df["label"], random_state=42
)

# 3. Dono models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}

results = []
trained = {}

for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)          # sirf train data pe fit
    preds = pipe.predict(X_test)        # test data pe evaluate
    trained[name] = pipe

    print("\n==", name, "==")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds), 4),
    })

    # 4. Confusion matrix image save
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"cm_{name.replace(' ', '_').lower()}.png", dpi=300)
    plt.close()

# 5. Metrics table
results_df = pd.DataFrame(results)
print("\n=== Final Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv("metrics.csv", index=False)

# 6. Best model (precision ke basis pe, kyunki false positives kam chahiye)
best_name = results_df.sort_values("Precision", ascending=False).iloc[0]["Model"]
print("\nBest model (by precision):", best_name)
joblib.dump(trained[best_name], "spam_pipeline.joblib")
print("Saved: spam_pipeline.joblib")