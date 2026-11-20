import requests
from transformers import AutoTokenizer


# Savunma sorusu E2: modelin tokenizer'ı "kitaplarımızdan" kelimesini nasıl parçalıyor?
# Ollama token'ları göstermez, yalnızca prompt_eval_count verir. Bu yüzden aynı temel modelin
# (Qwen3-1.7B) Hugging Face tokenizer'ı kullanılır ve sayılar Ollama ile çapraz doğrulanır.

HF_NAME = "Qwen/Qwen3-1.7B"       # Ollama'daki qwen3:1.7b ile aynı temel model
OLLAMA_MODEL = "qwen3:1.7b-q4_K_M"
OLLAMA_URL = "http://localhost:11434/api/generate"

tokenizer = AutoTokenizer.from_pretrained(HF_NAME)


def pieces(text):
    # Her token'ın metin karşılığı. Bir karakter birden çok baytsa ve token onu bölüyorsa
    # tek başına çözülemez, bu durumda bayt sayısı yazılır.
    result = []
    for token_id in tokenizer.encode(text, add_special_tokens=False):
        raw = tokenizer.convert_ids_to_tokens(token_id)
        decoded = tokenizer.decode([token_id])
        result.append(decoded if "�" not in decoded else f"<{raw}>")
    return result


def ollama_count(text):
    # raw=True: sohbet şablonu eklenmez, prompt_eval_count yalnızca bu metnin token sayısıdır
    payload = {"model": OLLAMA_MODEL, "prompt": text, "raw": True, "stream": False, "options": {"num_predict": 1}}
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["prompt_eval_count"]


print(f"Tokenizer: {HF_NAME} (vocab size {len(tokenizer)})\n")

# 1. E2: asıl soru
print("== kitaplarımızdan ==")
for text in ["kitaplarımızdan", " kitaplarımızdan"]:
    parts = pieces(text)
    print(f"{text!r}: {len(parts)} token -> {parts}")

# 2. Morfolojik yapı (Türkçe ekler): kitap + lar + ımız + dan = "kitaplar-ımız-dan"
print("\n== Aynı kök, farklı ekler ==")
for word in ["kitap", "kitaplar", "kitaplarımız", "kitaplarımızdan", "books", "our books", "from our books"]:
    print(f"{word!r:22} {len(pieces(word)):>2} token -> {pieces(word)}")

# 3. E1 için kanıt: aynı anlamı taşıyan Türkçe ve İngilizce cümleler
print("\n== Türkçe ve İngilizce karşılaştırma (token sayısı: HF tokenizer / Ollama prompt_eval_count) ==")
pairs = [
    ("One of our books is on the table.", "Kitaplarımızdan biri masanın üstünde."),
    ("The food was great but the service was very slow.", "Yemek harikaydı ama servis çok yavaştı."),
    ("The waiters were rude and the prices were too high.", "Garsonlar kabaydı ve fiyatlar çok yüksekti."),
]

total_en = total_tr = 0
for english, turkish in pairs:
    n_en, n_tr = len(pieces(english)), len(pieces(turkish))
    o_en, o_tr = ollama_count(english), ollama_count(turkish)
    total_en += n_en
    total_tr += n_tr
    print(f"EN {n_en:>2} / {o_en:>2}  {english}")
    print(f"TR {n_tr:>2} / {o_tr:>2}  {turkish}   (TR/EN = {n_tr / n_en:.2f}, karakter: {len(turkish)} vs {len(english)})")

print(f"\nToplam: EN {total_en} token, TR {total_tr} token, oran {total_tr / total_en:.2f}")
