
import requests

def generate_response(text):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "nlp-model-1",
        "prompt": text,
        "stream": False,
        "think": False
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()["response"]


text = "ben açım bana biraz paket yemek göster"

result = generate_response(text)

print(result)
