import json
from datetime import datetime


def save_campaign(campaign, output):
    record = {
        "timestamp": str(datetime.utcnow()),
        "campaign": campaign,
        "output": output
    }

    with open("data/campaigns.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
