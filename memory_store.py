import json
from datetime import datetime
import os

DATA_DIR = "data"
FILE_PATH = os.path.join(DATA_DIR, "campaigns.jsonl")

def save_campaign(campaign, output):
    # ✅ Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    record = {
        "timestamp": str(datetime.utcnow()),
        "campaign": campaign,
        "output": output
    }

    with open(FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
