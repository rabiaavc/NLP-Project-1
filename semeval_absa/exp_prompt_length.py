import argparse
import json
import random
import re

import requests

from model import JSON_SCHEMA, OLLAMA_URL


# Deney: few-shot örnek sayısı (prompt uzunluğu) arttıkça prefill ve decode süreleri nasıl değişir?
# Tek değişken k (few-shot örnek sayısı). Model, v2 prompt'u, parametreler ve test girdileri sabit.
#
# Kullanım:
#   python exp_prompt_length.py                                     (tam deney)
#   python exp_prompt_length.py --ks 0 5 --n-test 3 --output <dosya>  (küçük test)

MODEL = "absa-model-v2"                     # qwen3:1.7b Q4_K_M + v2 sistem prompt'u
MODELFILE = "modelfiles/Modelfile_v2"
TEST_PATH = "data/absa_120.json"
DEV_PATH = "data/absa_dev_80.json"
SEED = 42

OPTIONS = {"temperature": 0, "seed": SEED, "num_ctx": 8192}

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--ks", type=int, nargs="+", default=[0, 1, 5, 10, 20, 40])
arg_parser.add_argument("--n-test", type=int, default=30)
arg_parser.add_argument("--output", default="results/prompt_length_experiment.json")
args = arg_parser.parse_args()


def load(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# 1. v2 sistem prompt'unu Modelfile'dan oku (tek doğruluk kaynağı Modelfile)
with open(MODELFILE, "r", encoding="utf-8") as file:
    base_system = re.search(r'SYSTEM """(.*?)"""', file.read(), re.S).group(1).strip()


# 2. Few-shot örnekleri yalnızca dev setinden: positive ve negative sırayla, sabit karıştırma
dev = load(DEV_PATH)
positives = [r for r in dev if r["gold_sentiment"] == "positive"]
negatives = [r for r in dev if r["gold_sentiment"] == "negative"]
random.Random(SEED).shuffle(positives)
random.Random(SEED).shuffle(negatives)

# Sıra: pos, neg, pos, neg, ... k küçükken de örnekler dengeli olsun ve
# küçük k'nın örnekleri büyük k'nın alt kümesi olsun
shot_pool = [r for pair in zip(positives, negatives) for r in pair]
assert len(shot_pool) >= max(args.ks)


def build_system(k):
    if k == 0:
        return base_system

    lines = ["", "EXAMPLES:"]
    for r in shot_pool[:k]:
        answer = json.dumps({"sentiment": r["gold_sentiment"], "aspect": r["gold_aspect"]})
        lines.append(f'Review: {r["text"]}\nOutput: {answer}')

    return base_system + "\n" + "\n".join(lines)


# 3. Test girdileri: test setinden sabit, dengeli (yarısı positive, yarısı negative)
test = load(TEST_PATH)
test_pos = [r for r in test if r["gold_sentiment"] == "positive"]
test_neg = [r for r in test if r["gold_sentiment"] == "negative"]
random.Random(SEED).shuffle(test_pos)
random.Random(SEED).shuffle(test_neg)

half = args.n_test // 2
samples = test_pos[:half] + test_neg[:args.n_test - half]
assert not {r["text"] for r in samples} & {r["text"] for r in shot_pool}, "few-shot ve test çakışıyor"


def call(system, text, num_predict=None):
    options = dict(OPTIONS)
    if num_predict:
        options["num_predict"] = num_predict

    payload = {
        "model": MODEL,
        "system": system,
        "prompt": text,
        "stream": False,
        "think": False,
        "format": JSON_SCHEMA,
        "options": options,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def flush_prefix_cache():
    # Önceki k'nın prompt önbelleği (KV cache) sonraki k'nın ilk isteğini hızlandırmasın diye
    # model bellekten boşaltılıp aynı ayarlarla yeniden yüklenir. Bu bir ölçüm değildir,
    # sonuca yazılmaz. Yükleme süresi load_duration alanında ayrıdır, prefill'e karışmaz.
    requests.post(OLLAMA_URL, json={"model": MODEL, "keep_alive": 0}, timeout=60).raise_for_status()
    call("Reply with one word.", "Hi", num_predict=1)


print(f"Model: {MODEL} | k: {args.ks} | n_test: {len(samples)} | options: {OPTIONS}")

# 4. Isınma: model num_ctx=8192 ile belleğe yüklensin (yükleme süresi ölçüme girmesin)
call(build_system(0), "The soup was warm.")
print("Warm-up done")


records = []

for k in args.ks:
    system = build_system(k)
    flush_prefix_cache()

    for index, sample in enumerate(samples):
        data = call(system, sample["text"])

        try:
            parsed = json.loads(data["response"])
        except json.JSONDecodeError:
            parsed = None

        records.append({
            "k": k,
            "call_index": index,           # 0: ilk istek (prompt önbelleği boş), diğerleri: önbellek sıcak
            "id": sample["id"],
            "gold_sentiment": sample["gold_sentiment"],
            "gold_aspect": sample["gold_aspect"],
            "predicted_sentiment": parsed.get("sentiment") if isinstance(parsed, dict) else None,
            "predicted_aspect": parsed.get("aspect") if isinstance(parsed, dict) else None,
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration_ms": data.get("prompt_eval_duration", 0) / 1e6,
            "eval_count": data.get("eval_count"),
            "eval_duration_ms": data.get("eval_duration", 0) / 1e6,
            "load_duration_ms": data.get("load_duration", 0) / 1e6,
            "total_duration_ms": data.get("total_duration", 0) / 1e6,
        })

    print(f"k={k:>2} done | system prompt kelime: {len(system.split())} | ilk istek prompt token: {records[-len(samples)]['prompt_eval_count']}")

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump({"model": MODEL, "options": OPTIONS, "ks": args.ks, "n_test": len(samples), "records": records},
                  file, ensure_ascii=False, indent=2)

print(f"Results saved to {args.output}")
