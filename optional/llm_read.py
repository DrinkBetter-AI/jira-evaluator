import json

with open("project_categories.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for category, subcats in data.items():
    print(f"\n📁 {category}")
    for subcat, keys in subcats.items():
        print(f"  🔹 {subcat}: {', '.join(keys)}")