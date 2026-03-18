import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_NAMES = ["Negatif", "Nötr", "Pozitif"]
IMAGES_DIR = "images"

# ProsusAI/finbert label mapping: 0=positive, 1=negative, 2=neutral
# Project label mapping:          0=negative, 1=neutral,  2=positive
FINBERT_LABEL_MAP = {0: 2, 1: 0, 2: 1}


def load_data(filepath="all-data.csv"):
    df = pd.read_csv(
        filepath, header=None, names=["Sentiment", "Text"], encoding="ISO-8859-1"
    )
    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_finbert():
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        print("FinBERT modeli başarıyla yüklendi!")
        return tokenizer, model
    except Exception as e:
        print(f"FinBERT modeli yüklenirken hata oluştu: {e}")
        return None, None


def predict_with_finbert(texts, tokenizer, model, batch_size=16):
    if tokenizer is None or model is None:
        return None

    model.eval()
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**encoded).logits
            preds = torch.argmax(logits, dim=1).numpy()

        predictions.extend(FINBERT_LABEL_MAP[int(p)] for p in preds)

    return np.array(predictions)


def evaluate_model(y_true, y_pred):
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "report": classification_report(y_true, y_pred, target_names=LABEL_NAMES),
    }


def print_results(results, model_name):
    print(f"\n{model_name} Performansı:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print(f"\nDetaylı Sınıflandırma Raporu:\n{results['report']}")


def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    )
    plt.title(f"{model_name} — Karışıklık Matrisi")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    path = os.path.join(IMAGES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Kaydedildi: {path}")


def plot_model_comparison(all_results):
    models = list(all_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]

    x = np.arange(len(models))
    width = 0.18

    plt.figure(figsize=(12, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [all_results[m][metric] for m in models]
        plt.bar(x + (i - 1.5) * width, values, width, label=label, color=color)

    plt.xlabel("Model")
    plt.ylabel("Skor")
    plt.title("Model Performans Karşılaştırması")
    plt.xticks(x, models)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(IMAGES_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Kaydedildi: {path}")

    df = pd.DataFrame({
        "Model": models,
        "Accuracy": [f"{all_results[m]['accuracy']:.4f}" for m in models],
        "Precision": [f"{all_results[m]['precision']:.4f}" for m in models],
        "Recall": [f"{all_results[m]['recall']:.4f}" for m in models],
        "F1 Score": [f"{all_results[m]['f1']:.4f}" for m in models],
    })
    print(f"\n{df.to_string(index=False)}")


def cross_validate_smote(X, y, cv=5):
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=0)),
        ("clf", RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=0, n_jobs=-1,
        )),
    ])

    acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    f1 = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted")

    print(f"\nÇapraz Doğrulama ({cv}-fold, SMOTE Pipeline):")
    print(f"  Accuracy: {acc.mean()*100:.2f}% (±{acc.std()*100:.2f}%)")
    print(f"  F1 Score: {f1.mean()*100:.2f}% (±{f1.std()*100:.2f}%)")


def demo_predictions(texts, vectorizer, smote_clf, finbert_tok, finbert_mdl):
    label_map = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}

    print("\nÖrnek Tahminler:")
    print("-" * 70)
    for text in texts:
        vec = vectorizer.transform([clean_text(text)]).toarray()
        rf_pred = label_map[smote_clf.predict(vec)[0]]

        fb_pred = "—"
        if finbert_tok and finbert_mdl:
            fb_result = predict_with_finbert([text], finbert_tok, finbert_mdl)
            if fb_result is not None:
                fb_pred = label_map[fb_result[0]]

        print(f"  \"{text}\"")
        print(f"    RandomForest+SMOTE → {rf_pred}  |  FinBERT → {fb_pred}")
    print("-" * 70)


# ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # ── Veri yükleme ─────────────────────────────────────────────────
    print("Veri yükleniyor...")
    dataset = load_data()
    print(f"Veri kümesi boyutu: {dataset.shape}")
    print(f"\nİlk 5 satır:\n{dataset.head()}")
    print(f"\nDuyarlılık Dağılımı:\n{dataset['Sentiment'].value_counts()}")

    # ── Ön işleme ────────────────────────────────────────────────────
    print("\nMetin temizleme uygulanıyor...")
    dataset["Processed_Text"] = dataset["Text"].apply(clean_text)

    label_mapping = {"positive": 2, "neutral": 1, "negative": 0}
    dataset["Sentiment"] = dataset["Sentiment"].map(label_mapping)
    dataset = dataset.dropna()
    dataset["Processed_Text"] = dataset["Processed_Text"].astype(str)
    print(f"Temizleme sonrası boyut: {dataset.shape}")

    # ── TF-IDF ───────────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(dataset["Processed_Text"]).toarray()
    y = dataset["Sentiment"].values
    texts = dataset["Text"].tolist()

    # ── Train / test split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test, _, texts_test = train_test_split(
        X, y, texts, test_size=0.2, random_state=0, stratify=y,
    )

    all_results = {}

    # ── 1. Temel RandomForest ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. Temel RandomForest Modeli")
    print("=" * 60)

    basic_clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=0, n_jobs=-1,
    )
    basic_clf.fit(X_train, y_train)
    basic_results = evaluate_model(y_test, basic_clf.predict(X_test))
    all_results["Temel RF"] = basic_results
    print_results(basic_results, "Temel RandomForest")
    plot_confusion_matrix(basic_results["confusion_matrix"], "Temel RandomForest")

    # ── 2. SMOTE + RandomForest ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. SMOTE ile Dengelenmiş Model")
    print("=" * 60)

    smote = SMOTE(random_state=0)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  SMOTE öncesi: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"  SMOTE sonrası: {pd.Series(y_res).value_counts().to_dict()}")

    smote_clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=0, n_jobs=-1,
    )
    smote_clf.fit(X_res, y_res)
    smote_results = evaluate_model(y_test, smote_clf.predict(X_test))
    all_results["SMOTE RF"] = smote_results
    print_results(smote_results, "SMOTE RandomForest")
    plot_confusion_matrix(smote_results["confusion_matrix"], "SMOTE RandomForest")

    # ── 3. FinBERT ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. FinBERT Transformer Modeli")
    print("=" * 60)

    finbert_tok, finbert_mdl = load_finbert()

    if finbert_tok and finbert_mdl:
        print(f"Test seti üzerinde değerlendirme ({len(texts_test)} örnek)...")
        finbert_preds = predict_with_finbert(texts_test, finbert_tok, finbert_mdl)
        if finbert_preds is not None:
            finbert_results = evaluate_model(y_test, finbert_preds)
            all_results["FinBERT"] = finbert_results
            print_results(finbert_results, "FinBERT")
            plot_confusion_matrix(finbert_results["confusion_matrix"], "FinBERT")
    else:
        print("FinBERT modeli yüklenemedi — atlanıyor.")

    # ── Model karşılaştırması ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Model Karşılaştırması")
    print("=" * 60)
    plot_model_comparison(all_results)

    # ── Çapraz doğrulama ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Çapraz Doğrulama (SMOTE Pipeline)")
    print("=" * 60)
    cross_validate_smote(X, y)

    # ── Örnek tahminler ──────────────────────────────────────────────
    sample_texts = [
        "Company profits surge by 15% this quarter",
        "Stock market crashes amid economic uncertainty",
        "The company announced stable earnings for the fiscal year",
    ]
    demo_predictions(sample_texts, vectorizer, smote_clf, finbert_tok, finbert_mdl)


if __name__ == "__main__":
    main()
