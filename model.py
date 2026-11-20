
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_PS_URL = "http://localhost:11434/api/ps"

# Ollama "format" parametresi için JSON şeması
# Verildiğinde model yalnızca bu şemaya uyan çıktı üretebilir (constrained decoding)
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative"]},
        "aspect": {"type": "string"}
    },
    "required": ["sentiment", "aspect"]
}


# Gerekçe alanlı şema: model önce aspect'i, sonra kısa bir gerekçeyi, en son sentiment'i yazar
REASON_SCHEMA = {
    "type": "object",
    "properties": {
        "aspect": {"type": "string"},
        "reason": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "negative"]}
    },
    "required": ["aspect", "reason", "sentiment"]
}


def generate(text, model_name, use_schema=False, think=False, options=None, schema=None):

    # Dönüş değeri: (modelin cevabı, düşünme metni)
    # Düşünme metni yalnızca think=True olduğunda dolu gelir
    # schema verilirse JSON_SCHEMA yerine o kullanılır
    payload = {
        "model": model_name,
        "prompt": text,
        "stream": False,
        "think": think
    }

    if schema:
        payload["format"] = schema
    elif use_schema:
        payload["format"] = JSON_SCHEMA

    # Modelfile'daki parametreleri bu istek için değiştirir
    # Örnek: {"temperature": 0.7, "seed": 1}
    if options:
        payload["options"] = options

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    return data["response"], data.get("thinking", "")


def generate_response(text, model_name, use_schema=False):
    response, _ = generate(text, model_name, use_schema)
    return response


def get_loaded_model_memory(model_name):

    # Ollama'da bellekte yüklü olan modelin kapladığı alanı döndürür (byte)
    # Model yüklü değilse None döner
    response = requests.get(OLLAMA_PS_URL, timeout=10)
    response.raise_for_status()

    for model in response.json().get("models", []):
        if model["name"].split(":")[0] == model_name.split(":")[0]:
            return {"size": model["size"], "size_vram": model.get("size_vram", 0)}

    return None


if __name__ == "__main__":
    text = "The pizza was delicious."

    result = generate_response(text, "absa-model-v1")

    print(result)
