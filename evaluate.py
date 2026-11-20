
import json
import os
import re
import sys


LABELS = ["positive", "negative"]

# Olumsuzluk ifadeleri (not, no, never, can't, ...)
NEGATION_PATTERN = r"\b(not|no|never|nothing|cannot)\b|n't\b"

# Prompt'ta aspect'e eklenmemesi istenen tanımlık ve iyelikler
ARTICLE_PATTERN = r"^(the|a|an|our|their|my|his|her|its|your)\s"


def normalize(text):
    # Aspect karşılaştırması büyük/küçük harf ve boşluk farklarına duyarsız
    return " ".join(text.lower().split())


def is_grounded(aspect, text, case_sensitive):
    # Aspect metinde tam kelime(ler) olarak geçiyor mu?
    # Kelime sınırı kontrolü sayesinde "pizza", "pizzeria" içinde bulunmuş sayılmaz.
    if not aspect:
        return False

    pattern = r"(?<!\w)" + re.escape(aspect) + r"(?!\w)"
    flags = 0 if case_sensitive else re.IGNORECASE

    return re.search(pattern, text, flags) is not None


def get_prediction(record):
    # Geçerli yapıda olmayan çıktılar boş tahmin olarak sayılır
    if not record["valid_schema"]:
        return None, None

    predicted = record["parsed_response"]
    return predicted["sentiment"], predicted["aspect"]


def evaluate(results):

    total = len(results)
    metrics = {}

    # JSON Parsing
    metrics["JSON first-try success"] = sum(r["first_try_success"] for r in results) / total
    metrics["JSON parsed after cleaning"] = sum(r["parsed_response"] is not None for r in results) / total
    metrics["Valid schema"] = sum(r["valid_schema"] for r in results) / total

    # Parser hangi aşamada başarılı oldu? (eski sonuç dosyalarında bu alan yok)
    has_stage = all("parse_stage" in r for r in results)

    for stage in ["direct", "cleaned", "extracted", "failed"]:
        count = sum(r.get("parse_stage") == stage for r in results)
        metrics[f"Parse stage: {stage}"] = count / total if has_stage else float("nan")

    # Örnek başına ortalama süre (saniye)
    durations = [r["duration_seconds"] for r in results if "duration_seconds" in r]
    metrics["Avg seconds per sample"] = sum(durations) / len(durations) if durations else float("nan")

    # Sentiment Classification
    correct_sentiment = 0
    f1_scores = []

    for label in LABELS:
        tp = fp = fn = 0

        for record in results:
            predicted, _ = get_prediction(record)
            gold = record["gold_sentiment"]

            if predicted == label and gold == label:
                tp += 1
            elif predicted == label and gold != label:
                fp += 1
            elif predicted != label and gold == label:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"{label} precision"] = precision
        metrics[f"{label} recall"] = recall
        metrics[f"{label} F1"] = f1
        f1_scores.append(f1)

    for record in results:
        predicted, _ = get_prediction(record)
        if predicted == record["gold_sentiment"]:
            correct_sentiment += 1

    metrics["Sentiment accuracy"] = correct_sentiment / total
    metrics["Sentiment macro F1"] = sum(f1_scores) / len(f1_scores)

    # Aspect Extraction
    # Exact match: normalize edilmiş aspect'ler birebir aynı
    # Partial match: biri diğerini içeriyor (örnek: "pizza" ve "mushroom pizza")
    exact = 0
    partial = 0
    joint = 0
    predicted_count = 0

    for record in results:
        predicted_sentiment, predicted_aspect = get_prediction(record)

        if predicted_aspect is None:
            continue

        predicted_aspect = normalize(predicted_aspect)
        gold_aspect = normalize(record["gold_aspect"])

        # Boş olmayan her aspect bir tahmin sayılır (precision'ın paydası)
        if predicted_aspect != "":
            predicted_count += 1

        is_exact = predicted_aspect == gold_aspect
        is_partial = predicted_aspect != "" and (predicted_aspect in gold_aspect or gold_aspect in predicted_aspect)

        exact += is_exact
        partial += is_partial

        # Joint: hem sentiment hem aspect doğru
        if is_exact and predicted_sentiment == record["gold_sentiment"]:
            joint += 1

    metrics["Aspect exact match"] = exact / total
    metrics["Aspect partial match"] = partial / total

    # Entity extraction F1 (micro)
    # Her örnekte 1 gold aspect, model en fazla 1 aspect üretir.
    # TP: eşleşen tahmin, FP: eşleşmeyen tahmin, FN: bulunamayan gold aspect
    # Geçersiz veya boş çıktı tahmin sayılmaz: FN'yi artırır, FP'yi artırmaz.
    for match_name, true_positives in [("exact", exact), ("partial", partial)]:
        false_positives = predicted_count - true_positives
        false_negatives = total - true_positives

        precision = true_positives / (true_positives + false_positives) if predicted_count > 0 else 0
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"Aspect precision ({match_name})"] = precision
        metrics[f"Aspect recall ({match_name})"] = recall
        metrics[f"Aspect F1 ({match_name})"] = f1

    # Grounding: aspect kaynak metinden mi alınmış?
    # Case-sensitive: harfleriyle birebir kopyalanmış mı?
    # Case-insensitive: harf farkı yok sayılınca metinde var mı? (yoksa halüsinasyon)
    grounded_sensitive = 0
    grounded_insensitive = 0

    for record in results:
        _, predicted_aspect = get_prediction(record)

        grounded_sensitive += is_grounded(predicted_aspect, record["text"], case_sensitive=True)
        grounded_insensitive += is_grounded(predicted_aspect, record["text"], case_sensitive=False)

    metrics["Aspect grounding (case-sensitive)"] = grounded_sensitive / total
    metrics["Aspect grounding (case-insensitive)"] = grounded_insensitive / total
    metrics["Joint accuracy (sentiment + exact aspect)"] = joint / total

    return metrics


def print_confusion_matrix(name, results):

    # Satırlar gerçek etiket, sütunlar modelin tahmini
    # Geçerli yapıda olmayan çıktılar "invalid" sütununda sayılır
    columns = LABELS + ["invalid"]

    print(f"\n{name} - confusion matrix (rows: gold, columns: predicted)")
    print(f"{'':<12}" + "".join(f"{column:>10}" for column in columns))

    for gold in LABELS:
        counts = []

        for column in columns:
            count = 0

            for record in results:
                predicted, _ = get_prediction(record)
                predicted = predicted if predicted is not None else "invalid"

                if record["gold_sentiment"] == gold and predicted == column:
                    count += 1

            counts.append(count)

        print(f"{gold:<12}" + "".join(f"{count:>10}" for count in counts))


def print_negation_analysis(name, results):

    # Olumsuzluk ifadesi içeren ve içermeyen cümlelerde sentiment doğruluğu
    print(f"\n{name} - sentiment accuracy with and without negation words")

    for label in LABELS:
        with_negation = []
        without_negation = []

        for record in results:
            if record["gold_sentiment"] != label:
                continue

            if re.search(NEGATION_PATTERN, record["text"], re.IGNORECASE):
                with_negation.append(record)
            else:
                without_negation.append(record)

        for group_name, group in [("with negation", with_negation), ("without negation", without_negation)]:
            correct = sum(get_prediction(record)[0] == label for record in group)
            accuracy = correct / len(group) if group else 0

            print(f"  gold={label:<9} {group_name:<17} {correct:>3}/{len(group):<3} ({accuracy:.3f})")


def classify_aspect(record):

    # Aspect tahminini gold aspect ile karşılaştırarak hata türünü belirle
    _, predicted_aspect = get_prediction(record)

    if predicted_aspect is None:
        return "invalid"

    predicted = normalize(predicted_aspect)
    gold = normalize(record["gold_aspect"])

    if predicted == gold:
        return "exact"
    if predicted and predicted in gold:
        return "span too short"
    if gold in predicted:
        return "span too long"

    return "wrong target"


def print_aspect_analysis(name, results):

    # 1. Aspect hata türleri
    error_types = ["exact", "span too short", "span too long", "wrong target", "invalid"]
    counts = {error_type: 0 for error_type in error_types}

    for record in results:
        counts[classify_aspect(record)] += 1

    print(f"\n{name} - aspect error types")
    for error_type in error_types:
        print(f"  {error_type:<16} {counts[error_type]:>3}")

    # 2. Kural ihlalleri ve halüsinasyon
    # Not grounded: harf farkı yok sayılsa bile aspect metinde yok (halüsinasyon)
    # Capitalization: aspect metinde var ama harfleri korunmamış
    # Article: aspect "the", "our" gibi bir kelimeyle başlıyor
    violations = {"not grounded": [], "capitalization": [], "article": []}

    for record in results:
        _, predicted_aspect = get_prediction(record)

        if predicted_aspect is None:
            continue

        text = record["text"]

        if not is_grounded(predicted_aspect, text, case_sensitive=False):
            violations["not grounded"].append(record)
        elif not is_grounded(predicted_aspect, text, case_sensitive=True):
            violations["capitalization"].append(record)

        if re.match(ARTICLE_PATTERN, predicted_aspect, re.IGNORECASE):
            violations["article"].append(record)

    print(f"\n{name} - rule violations")
    for violation, records in violations.items():
        examples = ", ".join(f"[{r['id']}] {get_prediction(r)[1]!r}" for r in records[:5])
        print(f"  {violation:<16} {len(records):>3}  {examples}")


def print_errors(name, results, limit=10):

    print(f"\n{name} - errors")

    sentiment_errors = []
    aspect_errors = []

    for record in results:
        predicted_sentiment, predicted_aspect = get_prediction(record)

        if predicted_sentiment != record["gold_sentiment"]:
            sentiment_errors.append(record)

        if predicted_aspect is None or normalize(predicted_aspect) != normalize(record["gold_aspect"]):
            aspect_errors.append(record)

    print(f"Sentiment errors: {len(sentiment_errors)}")
    for record in sentiment_errors[:limit]:
        predicted_sentiment, _ = get_prediction(record)
        print(f"  [{record['id']}] gold={record['gold_sentiment']} predicted={predicted_sentiment} | {record['text']}")

    print(f"Aspect errors: {len(aspect_errors)}")
    for record in aspect_errors[:limit]:
        _, predicted_aspect = get_prediction(record)
        print(f"  [{record['id']}] gold={record['gold_aspect']!r} predicted={predicted_aspect!r} | {record['text']}")


def load_results(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# Bu bölüm yalnızca dosya doğrudan çalıştırıldığında devreye girer.
# Notebook'tan "import evaluate" yapıldığında yalnızca fonksiyonlar yüklenir.
if __name__ == "__main__":

    # Kullanım: python evaluate.py <sonuç_dosyası> [<sonuç_dosyası> ...]
    # Örnek:    python evaluate.py results/test/results_v2.json results/test/results_v4_reason.json
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <results_path> [<results_path> ...]")
        sys.exit(1)

    # 1. Sonuç dosyalarını oku ve değerlendir
    all_results = {}
    all_metrics = {}

    for path in sys.argv[1:]:
        name = os.path.basename(path)

        all_results[name] = load_results(path)
        all_metrics[name] = evaluate(all_results[name])

    # 2. Metrikleri yan yana yazdır
    names = list(all_metrics)
    metric_names = list(all_metrics[names[0]])

    print(f"\n{'Metric':<45}" + "".join(f"{name:>20}" for name in names))
    print("-" * (45 + 20 * len(names)))

    for metric in metric_names:
        print(f"{metric:<45}" + "".join(f"{all_metrics[name][metric]:>20.3f}" for name in names))

    print("\nSamples:", ", ".join(f"{name}={len(all_results[name])}" for name in names))

    # 3. Sentiment hata analizi
    for name in names:
        print_confusion_matrix(name, all_results[name])
        print_negation_analysis(name, all_results[name])

    # 4. Aspect hata analizi
    for name in names:
        print_aspect_analysis(name, all_results[name])

    # 5. Hatalı örnekleri göster
    for name in names:
        print_errors(name, all_results[name])
