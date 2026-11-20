
import argparse
import json
import time

from model import generate
from parser import parse_response


DATA_PATH = "data/absa_120.json"

ALLOWED_SENTIMENTS = ["positive", "negative"]

# Her aşama için ayrı JSON şeması
ASPECT_SCHEMA = {
    "type": "object",
    "properties": {"aspect": {"type": "string"}},
    "required": ["aspect"]
}

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": ALLOWED_SENTIMENTS}},
    "required": ["sentiment"]
}

STAGE_ORDER = ["direct", "cleaned", "extracted", "failed"]


# İki aşamalı yöntem:
#   1. aşama: aspect modeli yalnızca aspect'i çıkarır (ayrıntılı aspect kuralları)
#   2. aşama: sentiment modeli, cümle ve bulunan aspect verilince yalnızca duyguyu belirler (kısa prompt)
# Kullanım: python run_pipeline.py <çıktı_dosyası> [--data <veri_dosyası>]
# Örnek:    python run_pipeline.py results/dev/dev_results_pipeline.json --data data/absa_dev_80.json
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("output_path")
arg_parser.add_argument("--data", default=DATA_PATH, help="Veri dosyası (varsayılan: test seti)")
arg_parser.add_argument("--aspect-model", default="absa-stage-aspect")
arg_parser.add_argument("--sentiment-model", default="absa-stage-sentiment")
args = arg_parser.parse_args()

print(f"Aspect model: {args.aspect_model} | Sentiment model: {args.sentiment_model} | Data: {args.data}")


def sentiment_prompt(text, aspect):
    return f"Review: {text}\nAspect: {aspect}"


# 1. Veri setini oku
with open(args.data, "r", encoding="utf-8") as file:
    records = json.load(file)


# 2. Isınma (warm-up): iki modeli de belleğe yükle
generate("The soup was warm.", args.aspect_model, schema=ASPECT_SCHEMA)
generate(sentiment_prompt("The soup was warm.", "soup"), args.sentiment_model, schema=SENTIMENT_SCHEMA)
print("Warm-up done")


results = []

# 3. Her yorumu iki aşamada işle
for record in records:

    text = record["text"]
    start_time = time.perf_counter()

    # 1. aşama: aspect
    aspect_raw, _ = generate(text, args.aspect_model, schema=ASPECT_SCHEMA)
    aspect_parsed, aspect_first_try, aspect_stage = parse_response(aspect_raw)

    aspect = None
    if isinstance(aspect_parsed, dict) and isinstance(aspect_parsed.get("aspect"), str):
        aspect = aspect_parsed["aspect"]

    # 2. aşama: sentiment (aspect bulunamadıysa çalıştırılmaz)
    sentiment_raw = ""
    sentiment = None
    sentiment_first_try = False
    sentiment_stage = "failed"

    if aspect is not None:
        sentiment_raw, _ = generate(sentiment_prompt(text, aspect), args.sentiment_model, schema=SENTIMENT_SCHEMA)
        sentiment_parsed, sentiment_first_try, sentiment_stage = parse_response(sentiment_raw)

        if isinstance(sentiment_parsed, dict):
            sentiment = sentiment_parsed.get("sentiment")

    duration = time.perf_counter() - start_time

    # İki aşamanın sonucunu tek bir tahminde birleştir
    parsed_response = {"sentiment": sentiment, "aspect": aspect} if aspect is not None else None

    valid_schema = (
        parsed_response is not None
        and sentiment in ALLOWED_SENTIMENTS
        and isinstance(aspect, str)
    )

    aspect_in_text = valid_schema and aspect != "" and aspect in text

    # İki aşamadan daha kötü olan ayrıştırma aşaması kaydedilir
    parse_stage = max(aspect_stage, sentiment_stage, key=STAGE_ORDER.index)

    results.append({
        "id": record["id"],
        "text": text,
        "gold_sentiment": record["gold_sentiment"],
        "gold_aspect": record["gold_aspect"],
        "raw_response": {"aspect_stage": aspect_raw, "sentiment_stage": sentiment_raw},
        "parsed_response": parsed_response,
        "first_try_success": aspect_first_try and sentiment_first_try,
        "parse_stage": parse_stage,
        "valid_schema": valid_schema,
        "aspect_in_text": aspect_in_text,
        "model": f"{args.aspect_model} + {args.sentiment_model}",
        "used_schema": True,
        "duration_seconds": round(duration, 3)
    })

    # Her örnekten sonra sonuçları kaydet
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"Processed {len(results)}/{len(records)}")


print(f"Results saved to {args.output_path}")
