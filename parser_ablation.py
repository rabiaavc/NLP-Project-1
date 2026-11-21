import contextlib
import io
import json
import re

from evaluate import evaluate
from parser import parse_response


# A1 deneyi: parser.py'deki JSON çıkarma (extraction) satırı kaldırılırsa ne olur?
#
#   parsed_data = json.loads(cleaned_text[start:end + 1])
#
# Bu satır metindeki ilk "{" ile son "}" arasını alıp JSON olarak ayrıştırır.
# Model JSON'dan önce/sonra serbest metin yazdığında (v5_cot: önce gerekçe, sonra JSON)
# çıktıyı kurtaran tek aşama budur.
#
# Model yeniden çalıştırılmaz: kayıtlı 120 ham çıktı iki parser ile yeniden ayrıştırılır.
#   1) Tam parser: parser.py'deki parse_response (değiştirilmeden)
#   2) Extraction olmadan: aşağıdaki kopya; parser.py'nin 1-3. adımlarıyla aynı, 4. adım yok
#
# Kullanım: python parser_ablation.py

RESULTS_PATH = "results/test/results_v5_cot.json"
EXPECTED_KEYS = {"sentiment", "aspect"}
ALLOWED_SENTIMENTS = ["positive", "negative"]
EXAMPLE_ID = 1


def parse_without_extraction(raw_text):
    # parser.py ile aynı; yalnızca 4. adım (ilk "{" ile son "}" arasını alma) çıkarıldı

    # 1. İlk deneme: JSON'u doğrudan ayrıştır
    try:
        return json.loads(raw_text), True, "direct"
    except json.JSONDecodeError:
        pass

    # 2. Temizleme: <think> bloklarını ve Markdown işaretlerini kaldır
    cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    cleaned_text = cleaned_text.strip()

    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]

    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]

    cleaned_text = cleaned_text.strip()

    # 3. Temizlenen metni yeniden ayrıştır
    try:
        return json.loads(cleaned_text), False, "cleaned"
    except json.JSONDecodeError:
        pass

    # 4. (kaldırıldı) parsed_data = json.loads(cleaned_text[start:end + 1])
    return None, False, "failed"


def reparse(records, parse_function):
    # Kayıtlı ham çıktıları verilen parser ile yeniden ayrıştırır ve run.py'deki şema kontrolünü uygular
    new_records = []

    for record in records:
        # parser.py'nin her hatada yazdırdığı mesajlar tablo çıktısını kalabalıklaştırmasın
        with contextlib.redirect_stdout(io.StringIO()):
            parsed, first_try_success, stage = parse_function(record["raw_response"])

        valid_schema = (
            isinstance(parsed, dict)
            and set(parsed.keys()) == EXPECTED_KEYS
            and parsed["sentiment"] in ALLOWED_SENTIMENTS
            and isinstance(parsed["aspect"], str)
        )

        new_records.append({
            **record,
            "parsed_response": parsed,
            "first_try_success": first_try_success,
            "parse_stage": stage,
            "valid_schema": valid_schema,
        })

    return new_records


with open(RESULTS_PATH, "r", encoding="utf-8") as file:
    records = json.load(file)

full = reparse(records, parse_response)
ablated = reparse(records, parse_without_extraction)


# Doğrulama 1: tam parser, kayıtlı sonuç dosyasındaki ayrıştırmayı birebir yeniden üretmeli
assert all(
    new["parsed_response"] == old["parsed_response"]
    and new["parse_stage"] == old["parse_stage"]
    and new["valid_schema"] == old["valid_schema"]
    for new, old in zip(full, records)
), "Full parser does not reproduce the saved results"

# Doğrulama 2: kopya, extraction gerekmeyen her çıktıda tam parser ile aynı sonucu vermeli
assert all(
    a["parsed_response"] == f["parsed_response"] and a["parse_stage"] == f["parse_stage"]
    for f, a in zip(full, ablated) if f["parse_stage"] != "extracted"
), "Ablated parser differs from parser.py outside the extraction step"


# Sonuç tablosu
metrics_full = evaluate(full)
metrics_ablated = evaluate(ablated)

rows = [
    ("Raw json.loads success", "JSON first-try success"),
    ("Parsed (after cleaning)", "JSON parsed after cleaning"),
    ("Valid schema", "Valid schema"),
    ("Parse stage: extracted", "Parse stage: extracted"),
    ("Parse stage: failed", "Parse stage: failed"),
    ("Sentiment accuracy", "Sentiment accuracy"),
    ("Sentiment macro F1", "Sentiment macro F1"),
    ("Aspect F1 (exact)", "Aspect F1 (exact)"),
    ("Joint accuracy", "Joint accuracy (sentiment + exact aspect)"),
]

print(f"Results file: {RESULTS_PATH} ({len(records)} saved raw outputs, model not re-run)")
print("Checks passed: full parser reproduces saved results; ablated copy matches parser.py except step 4")
print()
print(f"{'metric':<26} {'full parser':>12} {'no extraction':>14}")
for label, key in rows:
    print(f"{label:<26} {metrics_full[key]:>12.3f} {metrics_ablated[key]:>14.3f}")


# Başarısız olan bir örnek
example_index = next(i for i, r in enumerate(records) if r["id"] == EXAMPLE_ID)
example = records[example_index]

print()
print(f"Example id {example['id']}: \"{example['text']}\"")
print(f"gold: {example['gold_sentiment']} / {example['gold_aspect']}")
print("Raw output:")
for line in example["raw_response"].strip().splitlines():
    print(f"  | {line}")
print()
print(f"full parser   -> stage: {full[example_index]['parse_stage']:<9} parsed: {full[example_index]['parsed_response']}")
print(f"no extraction -> stage: {ablated[example_index]['parse_stage']:<9} parsed: {ablated[example_index]['parsed_response']}"
      "  (counted as a miss: false negative for aspect, wrong for sentiment)")
