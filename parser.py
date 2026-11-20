
import json
import re


def parse_response(raw_text):

    # Dönüş değeri: (ayrıştırılmış veri, ilk denemede başarılı mı, başarılı olan aşama)
    # Aşamalar: "direct", "cleaned", "extracted", "failed"

    # 1. İlk deneme: JSON'u doğrudan ayrıştır
    try:
        parsed_data = json.loads(raw_text)
        return parsed_data, True, "direct"

    except json.JSONDecodeError:
        print("First JSON parsing failed!")

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
        parsed_data = json.loads(cleaned_text)
        return parsed_data, False, "cleaned"

    except json.JSONDecodeError:
        print("JSON parsing failed after cleaning!")

    # 4. Sohbet girişlerini kaldır: ilk "{" ile son "}" arasını al
    # Örnek: 'Here is the JSON: {"sentiment": "positive", "aspect": "pizza"}'
    start = cleaned_text.find("{")
    end = cleaned_text.rfind("}")

    if start != -1 and end > start:
        try:
            parsed_data = json.loads(cleaned_text[start:end + 1])
            return parsed_data, False, "extracted"

        except json.JSONDecodeError:
            pass

    print("JSON parsing failed after extracting the JSON object!")
    return None, False, "failed"


if __name__ == "__main__":
    examples = [
        '{"sentiment": "positive", "aspect": "pizza"}',
        '```json\n{"sentiment": "negative", "aspect": "staff"}\n```',
        'Here is the JSON: {"sentiment": "positive", "aspect": "service"} Hope this helps!',
        '<think>\n</think>\n\n{"sentiment": "negative", "aspect": "food"}',
        '{"sentiment": positive, "aspect": "pizza"}'
    ]

    for raw_text in examples:
        result, first_try_success, parse_stage = parse_response(raw_text)

        print("Parsed result:", result)
        print("First try success:", first_try_success)
        print("Parse stage:", parse_stage)
        print()
