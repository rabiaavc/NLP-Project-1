
import json
import random
from collections import Counter, defaultdict

from datasets import load_dataset


DATASET_NAME = "tomaarsen/setfit-absa-semeval-restaurants"
OUTPUT_PATH = "data/absa_120.json"
DEV_OUTPUT_PATH = "data/absa_dev_80.json"
EXCLUDED_PATH = "data/excluded_samples.json"

# Test seti: sonuçların raporlandığı set
# Dev seti: iyileştirme yöntemlerinin denenip seçildiği, test setiyle çakışmayan set
SAMPLES_PER_LABEL = 60
DEV_SAMPLES_PER_LABEL = 40
ALLOWED_LABELS = ["positive", "negative"]
SEED = 42


# 1. Veri setini oku
# Test bölümünde etiketler boş olduğu için train bölümü kullanılıyor.
dataset = load_dataset(DATASET_NAME, split="train")

print("Total rows:", len(dataset))


# 2. Aynı metne ait aspect'leri grupla
# Bir cümlede birden fazla aspect varsa aynı metin birden fazla satırda tekrarlanır.
grouped = defaultdict(list)

for row in dataset:
    grouped[row["text"]].append({
        "span": row["span"],
        "label": row["label"]
    })

print("Unique sentences:", len(grouped))


# 3. Yalnızca tek aspect içeren ve etiketi positive/negative olan cümleleri seç
candidates = defaultdict(list)

for text, aspects in grouped.items():

    if len(aspects) != 1:
        continue

    aspect = aspects[0]

    if aspect["label"] not in ALLOWED_LABELS:
        continue

    # Aspect metnin içinde birebir geçmeli
    if aspect["span"] not in text:
        continue

    candidates[aspect["label"]].append({
        "text": text,
        "gold_sentiment": aspect["label"],
        "gold_aspect": aspect["span"]
    })

for label in ALLOWED_LABELS:
    print(f"Single-aspect {label} candidates:", len(candidates[label]))


# 4. Elle incelenip elenen örnekleri oku
# Karışık duygu, ironi, örtük duygu, ikinci bir hedef veya anlamsız aspect içeren
# cümleler bu dosyada gerekçesiyle listeleniyor.
with open(EXCLUDED_PATH, "r", encoding="utf-8") as file:
    excluded = json.load(file)

excluded_texts = {record["text"] for record in excluded}

print("Manually excluded samples:", len(excluded_texts))
print("Exclusion reasons:", dict(Counter(record["reason"] for record in excluded)))


def build_records(selected, first_id):

    # Seçilen örnekleri karıştır ve numaralandır
    random.Random(SEED).shuffle(selected)

    records = []

    for index, record in enumerate(selected, start=first_id):
        records.append({
            "id": index,
            "text": record["text"],
            "gold_sentiment": record["gold_sentiment"],
            "gold_aspect": record["gold_aspect"],
            "word_count": len(record["text"].split())
        })

    return records


def validate_and_save(name, records, samples_per_label, output_path):

    texts = [record["text"] for record in records]
    label_counts = Counter(record["gold_sentiment"] for record in records)
    multi_word = sum(" " in record["gold_aspect"] for record in records)
    average_words = sum(record["word_count"] for record in records) / len(records)

    assert len(records) == samples_per_label * len(ALLOWED_LABELS)
    assert len(set(texts)) == len(texts), "Duplicate texts found"
    assert all(label_counts[label] == samples_per_label for label in ALLOWED_LABELS)
    assert all(record["gold_aspect"] in record["text"] for record in records)
    assert not any(text in excluded_texts for text in texts)

    print(f"\n{name} samples:", len(records))
    print("Unique texts:", len(set(texts)))
    print("Label distribution:", dict(label_counts))
    print("Multi-word aspects:", multi_word)
    print(f"Average word count: {average_words:.1f}")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    print("Saved to", output_path)


# 5. Her etiketten eşit sayıda örnek seç
# Havuz sabit seed ile karıştırılır ve elenmeyen örnekler sırayla alınır:
# ilk 60 örnek test setine, sonraki 40 örnek dev setine gider.
# Böylece bir örnek elendiğinde yalnızca yerine sıradaki aday gelir.
test_selected = []
dev_selected = []

for label in ALLOWED_LABELS:
    pool = sorted(candidates[label], key=lambda record: record["text"])
    random.Random(SEED).shuffle(pool)

    accepted = [record for record in pool if record["text"] not in excluded_texts]
    test_selected.extend(accepted[:SAMPLES_PER_LABEL])
    dev_selected.extend(accepted[SAMPLES_PER_LABEL:SAMPLES_PER_LABEL + DEV_SAMPLES_PER_LABEL])

# Test seti 1'den, dev seti 1001'den numaralandırılır
test_records = build_records(test_selected, first_id=1)
dev_records = build_records(dev_selected, first_id=1001)


# 6. Doğrula ve kaydet
validate_and_save("Test", test_records, SAMPLES_PER_LABEL, OUTPUT_PATH)
validate_and_save("Dev", dev_records, DEV_SAMPLES_PER_LABEL, DEV_OUTPUT_PATH)

# İki set arasında ortak cümle olmamalı
test_texts = {record["text"] for record in test_records}
assert not any(record["text"] in test_texts for record in dev_records), "Dev and test sets overlap"
print("\nNo overlap between test and dev sets")
