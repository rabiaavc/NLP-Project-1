import argparse
import json
import os
import textwrap
import time

from evaluate import is_grounded, normalize
from model import REASON_SCHEMA, generate
from parser import parse_response


# Canlı test (video) betiği
# Her örnek için ekrana sırasıyla şunlar yazdırılır:
#   1) modelin ham çıktısı, 2) json.loads / temizleme aşaması,
#   3) şema geçerliliği, 4) gold etiketle karşılaştırma
# run.py ile aynı fonksiyonları (generate, parse_response) ve evaluate.py ile aynı
# eşleştirme kuralını kullanır; böylece videodaki sonuçlar raporla aynı kodla üretilir.
#
# Kullanım:
#   python demo.py --data data/demo_samples.json --compare schema
#   python demo.py --data data/demo_samples.json --compare temperature
#   python demo.py --data data/live_samples.json --schema
#   python demo.py --text "The soup was cold." --gold negative --aspect soup --schema
#   python demo.py --data data/demo_samples.json --model absa-model-weak   (temizleme aşamasını göstermek için)

DEFAULT_MODEL = "absa-model-v4-reason"
ALLOWED_SENTIMENTS = ["positive", "negative"]
LINE = "-" * 78


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--data", help="Örnek dosyası (JSON listesi)")
arg_parser.add_argument("--text", help="Tek bir yeni cümle (dosya yerine)")
arg_parser.add_argument("--gold", choices=ALLOWED_SENTIMENTS, help="--text için beklenen sentiment")
arg_parser.add_argument("--aspect", help="--text için beklenen aspect")
arg_parser.add_argument("--model", default=DEFAULT_MODEL)
arg_parser.add_argument("--compare", choices=["none", "schema", "temperature"], default="none",
                        help="schema: format olmadan / formatla; temperature: farklı sıcaklık ve seed'ler")
arg_parser.add_argument("--schema", action="store_true", help="Ollama'ya JSON şeması gönder (compare=schema dışında)")
arg_parser.add_argument("--temperature", type=float, help="Tek ayarda temperature değeri")
arg_parser.add_argument("--temps", type=float, nargs="+", default=[0.0, 1.0], help="compare=temperature için sıcaklıklar")
arg_parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="compare=temperature için seed'ler")
args = arg_parser.parse_args()

if not args.data and not args.text:
    arg_parser.error("--data veya --text verilmeli")

# Model adında "reason" varsa çıktıda "reason" anahtarı da beklenir (run.py --reason ile aynı)
use_reason = "reason" in args.model
expected_keys = {"sentiment", "aspect", "reason"} if use_reason else {"sentiment", "aspect"}
reason_schema = REASON_SCHEMA if use_reason else None


# 1. Örnekleri oku
if args.text:
    records = [{"id": "new", "text": args.text, "gold_sentiment": args.gold, "gold_aspect": args.aspect}]
    data_name = "custom"
else:
    with open(args.data, "r", encoding="utf-8") as file:
        records = json.load(file)
    data_name = os.path.splitext(os.path.basename(args.data))[0]


# 2. Karşılaştırılacak ayarlar: (etiket, şema kullanılsın mı, Modelfile'ı geçersiz kılan seçenekler)
if args.compare == "schema":
    settings = [
        ("prompt-only (no format)", False, {}),
        ("schema-constrained (format)", True, {}),
    ]
elif args.compare == "temperature":
    # Modelfile'da seed sabit olduğu için temperature 1.0'da bile tek seed hep aynı çıktıyı verir.
    # Çeşitliliği görmek için her sıcaklık birden fazla seed ile çalıştırılır.
    settings = [
        (f"temperature {t} | seed {s}", args.schema, {"temperature": t, "seed": s})
        for t in args.temps for s in args.seeds
    ]
else:
    options = {"temperature": args.temperature} if args.temperature is not None else {}
    label = "schema-constrained (format)" if args.schema else "prompt-only (no format)"
    if options:
        label += f" | temperature {args.temperature}"
    settings = [(label, args.schema, options)]


def check_output(raw_response, record):
    # run.py ile aynı kontroller
    parsed, first_try_success, stage = parse_response(raw_response)

    valid_schema = (
        isinstance(parsed, dict)
        and set(parsed.keys()) == expected_keys
        and parsed["sentiment"] in ALLOWED_SENTIMENTS
        and isinstance(parsed["aspect"], str)
    )

    result = {
        "parsed_response": parsed,
        "first_try_success": first_try_success,
        "parse_stage": stage,
        "valid_schema": valid_schema,
        "sentiment_correct": None,
        "aspect_exact": None,
        "aspect_partial": None,
        "aspect_in_text": None,
    }

    if not valid_schema:
        return result

    predicted_aspect = normalize(parsed["aspect"])
    result["aspect_in_text"] = is_grounded(parsed["aspect"], record["text"], case_sensitive=True)

    if record.get("gold_sentiment"):
        result["sentiment_correct"] = parsed["sentiment"] == record["gold_sentiment"]

    if record.get("gold_aspect"):
        gold_aspect = normalize(record["gold_aspect"])
        result["aspect_exact"] = predicted_aspect == gold_aspect
        result["aspect_partial"] = predicted_aspect != "" and (
            predicted_aspect in gold_aspect or gold_aspect in predicted_aspect
        )

    return result


def mark(value):
    return "-" if value is None else ("✓" if value else "✗")


def print_raw(setting_label, raw_response):
    # Ham çıktı ayrıştırmadan önce yazdırılır; parser'ın hata mesajları bunun altında görünür
    print(f"  Setting: {setting_label}")
    print("  1) Raw output:")
    print(textwrap.indent(raw_response.strip() or "<empty>", "       | "))


def print_result(result, duration):
    parsed = result["parsed_response"]

    print(f"  2) json.loads on raw : {'OK' if result['first_try_success'] else 'FAILED'}"
          f"   -> parse stage: {result['parse_stage']}")
    print(f"  3) Schema valid      : {'YES' if result['valid_schema'] else 'NO'}")

    if result["valid_schema"]:
        print(f"  4) Sentiment         : {parsed['sentiment']:<9} {mark(result['sentiment_correct'])}")
        print(f"     Aspect            : \"{parsed['aspect']}\"   exact {mark(result['aspect_exact'])}"
              f" | partial {mark(result['aspect_partial'])} | in text {mark(result['aspect_in_text'])}")
    else:
        print("  4) Prediction        : none (invalid output counts as a miss)")

    print(f"     time: {duration:.2f} s")
    print()


print(f"Model: {args.model} | data: {args.data or 'custom text'} | compare: {args.compare}")
print(f"Expected keys: {sorted(expected_keys)}")


# 3. Isınma: ilk istek modeli belleğe yükler, süre ölçümünü bozmaması için kaydedilmez
generate("The soup was warm.", args.model, False, False, None, None)
print("Warm-up done")
print()


# 4. Her örneği her ayarla çalıştır
all_runs = []

for record in records:
    print(LINE)
    gold = f"{record.get('gold_sentiment') or '?'} / {record.get('gold_aspect') or '?'}"
    print(f"[id {record['id']}] \"{record['text']}\"")
    print(f"  gold: {gold}" + (f"   ({record['note']})" if record.get("note") else ""))
    print()

    sample_runs = []

    for setting_label, use_schema, options in settings:
        schema = reason_schema if use_schema else None

        start_time = time.perf_counter()
        raw_response, _ = generate(record["text"], args.model, use_schema, False, options, schema)
        duration = time.perf_counter() - start_time

        print_raw(setting_label, raw_response)
        result = check_output(raw_response, record)
        print_result(result, duration)

        run = {
            "id": record["id"],
            "text": record["text"],
            "gold_sentiment": record.get("gold_sentiment"),
            "gold_aspect": record.get("gold_aspect"),
            "setting": setting_label,
            "used_schema": use_schema,
            "options": options,
            "raw_response": raw_response,
            **result,
            "duration_seconds": round(duration, 3),
        }
        sample_runs.append(run)
        all_runs.append(run)

    # Aynı örnek için ayarlar arasında ham çıktı değişti mi?
    if args.compare == "schema":
        same = sample_runs[0]["raw_response"] == sample_runs[1]["raw_response"]
        print(f"  => Raw outputs identical with and without format: {'YES' if same else 'NO'}")
    elif args.compare == "temperature":
        for t in args.temps:
            outputs = {r["raw_response"] for r in sample_runs if r["options"]["temperature"] == t}
            print(f"  => temperature {t}: {len(args.seeds)} seeds -> {len(outputs)} different raw output(s)")
    print()


# 5. Ayar başına özet
print(LINE)
print("SUMMARY")
print(f"{'setting':<32} {'n':>3} {'raw JSON':>9} {'valid':>6} {'sent ✓':>7} {'asp ✓':>6} {'both ✓':>7}")

for setting_label, _, _ in settings:
    runs = [r for r in all_runs if r["setting"] == setting_label]
    n = len(runs)
    raw_ok = sum(r["first_try_success"] for r in runs)
    valid = sum(r["valid_schema"] for r in runs)
    sent = sum(bool(r["sentiment_correct"]) for r in runs)
    asp = sum(bool(r["aspect_exact"]) for r in runs)
    both = sum(bool(r["sentiment_correct"]) and bool(r["aspect_exact"]) for r in runs)
    print(f"{setting_label:<32} {n:>3} {raw_ok:>9} {valid:>6} {sent:>7} {asp:>6} {both:>7}")


# 6. Kanıt olarak kaydet
os.makedirs("results/live", exist_ok=True)
output_path = f"results/live/{data_name}_{args.compare}_{args.model}.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(all_runs, file, ensure_ascii=False, indent=2)

print()
print(f"Saved to {output_path}")
