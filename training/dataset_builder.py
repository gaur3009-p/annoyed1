import json

INPUT_FILE = "data/campaigns.jsonl"
OUTPUT_FILE = "data/train_global.jsonl"

def build_dataset():
    with open(INPUT_FILE, "r") as f, open(OUTPUT_FILE, "w") as out:
        for line in f:
            record = json.loads(line)

            instruction = f"""
Create a marketing campaign.

Brand: {record['campaign']['brand']}
Industry: {record['campaign']['industry']}
Audience: {record['campaign']['audience']}
Goal: {record['campaign']['goal']}
Tone: {record['campaign']['tone']}
"""

            sample = {
                "instruction": instruction.strip(),
                "response": record["output"].strip()
            }

            out.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    build_dataset()
