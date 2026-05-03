import pandas as pd
import openai
import json
import time

client = openai.OpenAI(api_key="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")
model = "gpt-4o"
EPIC_KEY='MB-4102'
df = pd.read_csv(f"jira_epic_tickets_extended_{EPIC_KEY}.csv")

def clean(text):
    if not isinstance(text, str):
        return ""
    return text.replace("\n", " ").replace("\r", " ").strip()

tasks = [
    {
        "key": row["Key"],
        "summary": clean(row["Summary"]),
        "description": clean(row["Description"])
    }
    for _, row in df.iterrows()
]

def ask_gpt_chunk(chunk_tasks):
    system_msg = (
        "You are a project planner assistant. Group the following Jira tickets into categories and subcategories "
        "based on their purpose or topic. Respond only with valid JSON in this format:\n"
        "{\n"
        "  'Category A': {\n"
        "    'Subcategory A1': ['MB-123', 'MB-124']\n"
        "  }\n"
        "}"
    )

    user_msg = "Here is a list of Jira tickets:\n"
    for task in chunk_tasks:
        user_msg += f"- {task['key']}: {task['summary']} — {task['description']}\n"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )

        raw_content = response.choices[0].message.content.strip()


        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1].strip()
            if raw_content.startswith("json"):
                raw_content = raw_content[len("json"):].strip()

        if not raw_content:
            print("⚠️ Empty response from GPT.")
            return {}

        return json.loads(raw_content)

    except json.JSONDecodeError as je:
        print("❌ JSON decode error:", je)
        print("↪️ Raw content was:\n", raw_content[:500])
        return {}

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {}

def merge_categories(all_chunks):
    merged = {}
    for chunk in all_chunks:
        for category, subcats in chunk.items():
            if category not in merged:
                merged[category] = {}
            for subcat, keys in subcats.items():
                merged[category].setdefault(subcat, [])
                merged[category][subcat].extend(keys)
                merged[category][subcat] = list(set(merged[category][subcat]))  # deduplicate
    return merged

chunk_size = 20
chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

all_results = []
print(f"📦 Sending {len(chunks)} chunks to GPT-4o...")

for idx, chunk in enumerate(chunks):
    print(f"🔁 Processing chunk {idx + 1}/{len(chunks)}")
    result = ask_gpt_chunk(chunk)
    if result:
        all_results.append(result)
    time.sleep(2)

final_structure = merge_categories(all_results)

with open(f"project_categories_{EPIC_KEY}.json", "w", encoding="utf-8") as f:
    json.dump(final_structure, f, indent=2, ensure_ascii=False)

print("✅ All tasks categorized. Saved to project_categories.json")