
import json
from google.cloud import firestore

PROJECT = "astrohealth-509609"
db = firestore.Client(project=PROJECT)

report = {
    "project": PROJECT,
    "collections": []
}

def get_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "map"
    if value is None:
        return "null"
    return type(value).__name__

def inspect(collection, path, depth=0):
    if depth > 4:
        return

    entry = {
        "path": path,
        "sampled_documents": 0,
        "fields": {},
        "subcollections": set()
    }

    report["collections"].append(entry)

    try:
        for doc in collection.limit(50).stream():
            entry["sampled_documents"] += 1

            for key, value in doc.to_dict().items():
                entry["fields"].setdefault(key, set()).add(
                    get_type(value)
                )

            for sub in doc.reference.collections():
                entry["subcollections"].add(sub.id)

                inspect(
                    sub,
                    path + "/{document}/" + sub.id,
                    depth + 1
                )

        entry["fields"] = {
            k: sorted(v) for k, v in entry["fields"].items()
        }
        entry["subcollections"] = sorted(entry["subcollections"])

    except Exception as e:
        entry["error"] = type(e).__name__

try:
    for collection in db.collections():
        inspect(collection, collection.id)

    with open("final_data_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Audit complete.")
    print("Collections:", len(report["collections"]))
    print("Report: final_data_audit.json")

except Exception as e:
    print("Audit failed:", type(e).__name__)
