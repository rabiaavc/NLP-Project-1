import json
import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Savunma sorusu G1/G2: klasik model kaç etiketli örnekle LLM'e yaklaşır?
# Eğitim havuzu: test ve dev setinde OLMAYAN, elenmemiş, tek aspect'li cümleler.
# prepare_data.py ile aynı seçim mantığı kullanılır (aynı seed ve sıralama).
# Test: data/absa_120.json (değişmez). Havuz ile test arasında ortak cümle yoktur.

DATASET_NAME = "tomaarsen/setfit-absa-semeval-restaurants"
SEED = 42
LABELS = ["positive", "negative"]
TRAIN_SIZES = [10, 20, 50, 100, 200, 300]     # dengeli (yarısı positive, yarısı negative)
N_DRAWS = 10                                  # her boyut için rastgele eğitim alt kümesi sayısı
LLM_ACCURACY = {"LLM v2 (no training)": 0.942, "LLM v4_reason (no training)": 0.992}

INK, MUTED, GRID = "#1f2933", "#6b7785", "#e3e7ec"
BLUE, ORANGE, GREEN = "#2a6fdb", "#c2571a", "#2f8f5b"


def load(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


test = load("data/absa_120.json")
dev = load("data/absa_dev_80.json")
excluded_texts = {r["text"] for r in load("data/excluded_samples.json")}

# 1. Aday cümleler (prepare_data.py adım 2-3 ile aynı)
grouped = defaultdict(list)
for row in load_dataset(DATASET_NAME, split="train"):
    grouped[row["text"]].append({"span": row["span"], "label": row["label"]})

candidates = {label: [] for label in LABELS}
for text, aspects in grouped.items():
    if len(aspects) == 1 and aspects[0]["label"] in LABELS and aspects[0]["span"] in text:
        candidates[aspects[0]["label"]].append({"text": text, "label": aspects[0]["label"]})

# 2. Havuz: elenmeyenlerden test (60) ve dev (40) için ayrılanlar çıkarılır, kalan eğitim havuzudur
pool = {}
for label in LABELS:
    ordered = sorted(candidates[label], key=lambda r: r["text"])
    random.Random(SEED).shuffle(ordered)
    accepted = [r for r in ordered if r["text"] not in excluded_texts]
    pool[label] = accepted[100:]            # ilk 60: test, sonraki 40: dev

used = {r["text"] for r in test} | {r["text"] for r in dev}
assert not any(r["text"] in used for label in LABELS for r in pool[label]), "havuz test/dev ile çakışıyor"
print("Kalan havuz (elenmemiş, test/dev dışı):", {label: len(pool[label]) for label in LABELS})

# Kalan havuzda yalnızca 6 negative var (negatiflerin çoğu test, dev veya elenenlere gitti).
# Dengeli eğitim için havuz iki kaynakla genişletilir. İkisi de test setinde yoktur:
#  - dev seti (yöntem seçimi için kullanıldı, test ile ortak cümle yok)
#  - elenen cümleler: değerlendirme için fazla belirsizdi ama SemEval etiketleri geçerli.
#    Etiketi veya metni şüpheli olanlar (label_mismatch, incomplete, bad_aspect) kullanılmaz.
NOISY_REASONS = {"label_mismatch", "incomplete", "bad_aspect"}
test_texts_set = {r["text"] for r in test}

for record in dev:
    pool[record["gold_sentiment"]].append({"text": record["text"], "label": record["gold_sentiment"]})

excluded_used = 0
for record in load("data/excluded_samples.json"):
    if record["reason"] not in NOISY_REASONS:
        pool[record["gold_sentiment"]].append({"text": record["text"], "label": record["gold_sentiment"]})
        excluded_used += 1

for label in LABELS:                                   # aynı cümle iki kez girmesin
    unique = {r["text"]: r for r in pool[label]}
    pool[label] = list(unique.values())

assert not any(r["text"] in test_texts_set for label in LABELS for r in pool[label]), "eğitim havuzu test ile çakışıyor"
print(f"Eğitim havuzu (dev + {excluded_used} elenen cümle eklendi):", {label: len(pool[label]) for label in LABELS},
      "| test ile ortak cümle yok")

max_balanced = 2 * min(len(pool[label]) for label in LABELS)
sizes = [s for s in TRAIN_SIZES if s <= max_balanced] + [max_balanced]
sizes = sorted(set(sizes))

test_texts = [r["text"] for r in test]
test_labels = [r["gold_sentiment"] for r in test]


def evaluate_subset(make_vectorizer, subset):
    vectorizer = make_vectorizer()
    X_train = vectorizer.fit_transform([r["text"] for r in subset])
    model = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train, [r["label"] for r in subset])
    return accuracy_score(test_labels, model.predict(vectorizer.transform(test_texts)))


methods = {"BoW + LR": lambda: CountVectorizer(lowercase=True), "TF-IDF + LR": lambda: TfidfVectorizer(lowercase=True)}
results = {name: {} for name in methods}

# 3. Her boyut için N_DRAWS farklı dengeli eğitim alt kümesi
print(f"\n{'train':>6} " + "".join(f"{name:>26}" for name in methods))
for size in sizes:
    per_class = size // 2
    row = []
    for name, make_vectorizer in methods.items():
        accuracies = []
        for draw in range(N_DRAWS if size < max_balanced else 1):
            rng = random.Random(draw)
            subset = rng.sample(pool["positive"], per_class) + rng.sample(pool["negative"], per_class)
            accuracies.append(evaluate_subset(make_vectorizer, subset))
        results[name][size] = accuracies
        row.append(f"{np.mean(accuracies):.3f} (min {min(accuracies):.3f}, max {max(accuracies):.3f})")
    print(f"{size:>6} " + "".join(f"{cell:>26}" for cell in row))

# 4. Hız: en büyük dengeli eğitimle fit süresi ve örnek başına tahmin süresi
subset = pool["positive"][:max_balanced // 2] + pool["negative"][:max_balanced // 2]
vectorizer = TfidfVectorizer(lowercase=True)
start = time.perf_counter()
X_train = vectorizer.fit_transform([r["text"] for r in subset])
model = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train, [r["label"] for r in subset])
fit_seconds = time.perf_counter() - start

start = time.perf_counter()
for _ in range(20):
    model.predict(vectorizer.transform(test_texts))
predict_ms = (time.perf_counter() - start) / 20 / len(test_texts) * 1000
print(f"\nTF-IDF + LR, {max_balanced} örnek: eğitim {fit_seconds * 1000:.0f} ms, tahmin {predict_ms:.3f} ms/örnek (LLM: 700-1900 ms/örnek)")

with open("results/learning_curve_tfidf.json", "w", encoding="utf-8") as file:
    json.dump({"pool": {label: len(pool[label]) for label in LABELS}, "sizes": sizes, "n_draws": N_DRAWS,
               "accuracies": {name: {str(s): v for s, v in per_size.items()} for name, per_size in results.items()},
               "llm": LLM_ACCURACY, "fit_seconds": fit_seconds, "predict_ms_per_sample": predict_ms},
              file, indent=2)

# 5. Grafik
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.5, 4.6))

for name, color in [("TF-IDF + LR", BLUE), ("BoW + LR", ORANGE)]:
    xs = list(results[name])
    means = [np.mean(results[name][s]) for s in xs]
    lows = [min(results[name][s]) for s in xs]
    highs = [max(results[name][s]) for s in xs]
    ax.fill_between(xs, lows, highs, color=color, alpha=0.12, linewidth=0)
    ax.plot(xs, means, color=color, linewidth=2, marker="o", markersize=6)
    shift = 0.012 if name == "TF-IDF + LR" else -0.012          # iki etiket üst üste binmesin
    ax.text(xs[-1] * 1.04, means[-1] + shift, name, color=color, va="center", fontsize=9)

for (name, value), color in zip(LLM_ACCURACY.items(), [MUTED, GREEN]):
    ax.axhline(value, color=color, linewidth=1.2, linestyle="--")
    ax.text(sizes[0], value + 0.008, f"{name}: {value:.3f}", color=color, fontsize=9)

ax.set_xscale("log")
ax.set_xticks(sizes)
ax.set_xticklabels([str(s) for s in sizes])
ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 1.9)
ax.set_ylim(0.45, 1.03)
ax.set_xlabel("Labeled training examples (balanced)")
ax.set_ylabel("Sentiment accuracy on the 120 test samples")
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.set_title("Classical baseline learning curve vs LLM", loc="left", fontsize=11, fontweight="bold", pad=22)
ax.text(0, 1.02, "Band: min-max over 10 random subsets. Training pool = dev set + excluded (harder) sentences.",
        transform=ax.transAxes, fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig("results/learning_curve_tfidf.png", dpi=150, facecolor="white")
print("saved results/learning_curve_tfidf.png")
