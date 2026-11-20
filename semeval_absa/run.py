
import argparse
import json
import time

from model import REASON_SCHEMA, generate, get_loaded_model_memory
from parser import parse_response


DATA_PATH = "data/absa_120.json"

ALLOWED_SENTIMENTS = ["positive", "negative"]
EXPECTED_KEYS = {"sentiment", "aspect"}


# Kullanım: python run.py <model_adı> <çıktı_dosyası> [seçenekler]
# Örnekler:
#   python run.py absa-model-v1 results/test/results_v1.json
#   python run.py absa-model-v2 results/test/results_v2_schema.json --schema
#   python run.py absa-model-v2 results/test/results_v2_temp0.7_seed1.json --temperature 0.7 --seed 1
#   python run.py absa-model-v2 results/test/results_v2_think.json --think --num-predict 2048
#   python run.py absa-model-v2 results/dev/dev_results_v2.json --data data/absa_dev_80.json
#   python run.py absa-model-v4-reason results/test/results_v4_reason.json --schema --reason
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("model_name")
arg_parser.add_argument("output_path")
arg_parser.add_argument("--data", default=DATA_PATH, help="Veri dosyası (varsayılan: test seti)")
arg_parser.add_argument("--schema", action="store_true", help="Ollama'ya JSON şeması gönder (constrained decoding)")
arg_parser.add_argument("--reason", action="store_true", help="Çıktıda 'reason' alanı bekle ve gerekçe şemasını kullan")
arg_parser.add_argument("--think", action="store_true", help="Qwen3 düşünme modunu aç")
arg_parser.add_argument("--temperature", type=float, help="Modelfile'daki temperature değerini değiştir")
arg_parser.add_argument("--seed", type=int, help="Modelfile'daki seed değerini değiştir")
arg_parser.add_argument("--num-predict", type=int, help="Modelfile'daki num_predict değerini değiştir")
args = arg_parser.parse_args()

model_name = args.model_name
output_path = args.output_path

# Yalnızca komutta verilen parametreler Modelfile'ı geçersiz kılar
options = {}
if args.temperature is not None:
    options["temperature"] = args.temperature
if args.seed is not None:
    options["seed"] = args.seed
if args.num_predict is not None:
    options["num_predict"] = args.num_predict

# --reason verilirse çıktıda "reason" anahtarı da beklenir
expected_keys = EXPECTED_KEYS | {"reason"} if args.reason else EXPECTED_KEYS
schema = REASON_SCHEMA if args.reason and args.schema else None

print(f"Model: {model_name} | Data: {args.data} | JSON schema: {args.schema} | reason: {args.reason} | think: {args.think} | options: {options}")


# 1. Veri setini oku
with open(args.data, "r", encoding="utf-8") as file:
    records = json.load(file)


# 2. Isınma (warm-up)
# İlk istek modeli belleğe yüklediği için yavaştır. Süre ölçümünü bozmaması için
# veri setinde olmayan bir cümleyle bir kez çağırıp sonucu kaydetmiyoruz.
generate("The soup was warm.", model_name, args.schema, args.think, options, schema)
print("Warm-up done")


results = []

# 3. Her yorumu modele gönder
for record in records:

    text = record["text"]

    # Modelden ham yanıt al ve süresini ölç
    start_time = time.perf_counter()
    raw_response, thinking = generate(text, model_name, args.schema, args.think, options, schema)
    duration = time.perf_counter() - start_time

    # JSON ayrıştırma ve temizleme
    parsed_response, first_try_success, parse_stage = parse_response(raw_response)

    # Çıktının beklenen yapıda olup olmadığını kontrol et
    valid_schema = (
        isinstance(parsed_response, dict)
        and set(parsed_response.keys()) == expected_keys
        and parsed_response["sentiment"] in ALLOWED_SENTIMENTS
        and isinstance(parsed_response["aspect"], str)
    )

    # Çıkarılan aspect metinde birebir geçiyor mu?
    aspect_in_text = (
        valid_schema
        and parsed_response["aspect"] != ""
        and parsed_response["aspect"] in text
    )

    # Sonucu kaydet
    results.append({
        "id": record["id"],
        "text": text,
        "gold_sentiment": record["gold_sentiment"],
        "gold_aspect": record["gold_aspect"],
        "raw_response": raw_response,
        "thinking": thinking,
        "parsed_response": parsed_response,
        "first_try_success": first_try_success,
        "parse_stage": parse_stage,
        "valid_schema": valid_schema,
        "aspect_in_text": aspect_in_text,
        "model": model_name,
        "used_schema": args.schema,
        "think": args.think,
        "options": options,
        "duration_seconds": round(duration, 3)
    })

    # Her örnekten sonra sonuçları kaydet
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"Processed {len(results)}/{len(records)}")


print(f"Results saved to {output_path}")


# 4. Modelin bellekte kapladığı alan
memory = get_loaded_model_memory(model_name)

if memory:
    print(f"Model memory: {memory['size'] / 1024 ** 3:.2f} GB (on GPU: {memory['size_vram'] / 1024 ** 3:.2f} GB)")
else:
    print("Model memory: model is not loaded")
