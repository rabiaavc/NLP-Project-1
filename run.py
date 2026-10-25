
import json

with open("data/project_dataset_120.json", "r", encoding="utf-8") as file:
    records = json.load(file)

print("Total records:", len(records))
print("First text:", records[0]["text"])
