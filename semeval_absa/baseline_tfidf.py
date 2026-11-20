import json
import random

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# Klasik temel çizgi: TF-IDF / Bag-of-Words + Logistic Regression
# Yalnızca sentiment sınıflandırması yapılır. Aspect çıkarma bu yöntemle yapılmaz
# (kelime torbası modelleri cümleden bir ifade kopyalayamaz).
# Eğitim: dev seti (80 örnek). Test: test seti (120 örnek). İki setin ortak cümlesi yok.

TRAIN_PATH = "data/absa_dev_80.json"
TEST_PATH = "data/absa_120.json"
SEED = 42


def load(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


train = load(TRAIN_PATH)
test = load(TEST_PATH)

test_texts = [r["text"] for r in test]
test_labels = [r["gold_sentiment"] for r in test]


def run(name, vectorizer, train_records):
    texts = [r["text"] for r in train_records]
    labels = [r["gold_sentiment"] for r in train_records]

    X_train = vectorizer.fit_transform(texts)
    X_test = vectorizer.transform(test_texts)

    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X_train, labels)
    predicted = model.predict(X_test)

    accuracy = accuracy_score(test_labels, predicted)
    macro_f1 = f1_score(test_labels, predicted, average="macro")
    print(f"{name:<46} n_train={len(train_records):>3}  accuracy={accuracy:.3f}  macro F1={macro_f1:.3f}")

    return model, vectorizer, accuracy


print("Test seti:", len(test), "örnek (LLM v4_reason sentiment accuracy: 0.992, v2: 0.942)\n")

# 1. BoW ve TF-IDF karşılaştırması (80 eğitim örneği)
bow_model, bow_vec, _ = run("Bag-of-Words + LR", CountVectorizer(lowercase=True), train)
tfidf_model, tfidf_vec, _ = run("TF-IDF + LR", TfidfVectorizer(lowercase=True), train)
run("TF-IDF (1-2 gram) + LR", TfidfVectorizer(lowercase=True, ngram_range=(1, 2)), train)


# 2. Çok az etiketli örnekle (10 örnek: 5 positive + 5 negative), 5 farklı seçim
print("\n10 etiketli örnekle TF-IDF + LR (5 farklı rastgele seçim):")

positives = [r for r in train if r["gold_sentiment"] == "positive"]
negatives = [r for r in train if r["gold_sentiment"] == "negative"]
accuracies = []

for seed in range(1, 6):
    rng = random.Random(seed)
    subset = rng.sample(positives, 5) + rng.sample(negatives, 5)
    _, _, accuracy = run(f"  seed {seed}", TfidfVectorizer(lowercase=True), subset)
    accuracies.append(accuracy)

print(f"  ortalama accuracy={np.mean(accuracies):.3f}  min={min(accuracies):.3f}  max={max(accuracies):.3f}")


# 3. Modelin en çok ağırlık verdiği kelimeler (TF-IDF, 80 örnek)
# Katsayı pozitifse "positive" sınıfını, negatifse "negative" sınıfını destekler
words = np.array(tfidf_vec.get_feature_names_out())
order = np.argsort(tfidf_model.coef_[0])
classes = tfidf_model.classes_
print(f"\nSınıflar: {list(classes)}  (katsayı > 0: {classes[1]})")
print("En güçlü 'negative' kelimeleri:", ", ".join(words[order[:10]]))
print("En güçlü 'positive' kelimeleri:", ", ".join(words[order[::-1][:10]]))
