
import json

def parse_response(raw_text):

    # 1. İlk deneme: JSON'u doğrudan ayrıştır
    try:
        parsed_data = json.loads(raw_text)
        return parsed_data, True

    except json.JSONDecodeError:
        print("First JSON parsing failed!")

    # 2. Temizleme: Markdown işaretlerini kaldır
    cleaned_text = raw_text.strip()

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
        return parsed_data, False

    except json.JSONDecodeError:
        print("JSON parsing failed after cleaning!")
        return None, False


if __name__ == "__main__":
    raw_text = '{"intent": takeaway_order, "slots": []}'

    result, first_try_success = parse_response(raw_text)

    print("Parsed result:", result)
    print("First try success:", first_try_success)

