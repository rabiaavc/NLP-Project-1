
import json

from model import generate_response
from parser import parse_response


ALLOWED_INTENTS = [
    "calendar_set",
    "play_music",
    "takeaway_order",
    "transport_taxi",
    "transport_ticket",
    "weather_query"
]


# 1. Veri setini oku
with open("data/project_dataset_120.json", "r", encoding="utf-8") as file:
    records = json.load(file)


results = []

# 2. İlk 3 metni işle
for record in records:

    text = record["text"]

    # Modelden ham yanıt al
    raw_response = generate_response(text)

    # JSON ayrıştırma ve temizleme
    parsed_response, first_try_success = parse_response(raw_response)


    valid_intent = (
        isinstance(parsed_response, dict)
        and parsed_response.get("intent") in ALLOWED_INTENTS
    )


    # Sonucu kaydet
    results.append({
        "id": record["id"],
        "text": text,
        "gold_intent": record["gold_intent"],
        "gold_slots": record["gold_slots"],
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "first_try_success": first_try_success,
        "valid_intent": valid_intent
    })

    
    # Her örnekten sonra sonuçları kaydet
    with open("results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"Processed {len(results)}/{len(records)}")


print("Results saved to results.json")
