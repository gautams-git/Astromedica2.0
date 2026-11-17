
import json
from google.cloud import firestore

PROJECT_ID = "astrohealth-509609"
MAX_DOCS = 100

db = firestore.Client(project=PROJECT_ID)

report = {
    "project": PROJECT_ID,
    "schema": []
}

def inspect_collection(collection, parent=""):
    path = f"{parent}/{collection.id}" if parent else collection.id

    entry = {
        "collection": path,
        "sample_count": 0,
        "field_names": set(),
        "field_types": {},
        "subcollections": set()
    }

    try:
        for snapshot in collection.limit(MAX_DOCS).stream():

            entry["sample_count"] += 1

            # Only inspect field names and data types.
            data = snapshot.to_dict() or {}

            for key, value in data.items():
                entry["field_names"].add(key)

                if isinstance(value, bool):
                    kind = "boolean"
                elif isinstance(value, str):
                    kind = "string"
                elif isinstance(value, (int, float)):
                    kind = "number"
                elif isinstance(value, list):
                    kind = "array"
                elif isinstance(value, dict):
                    kind = "map"
                else:
                    kind = type(value).__name__

                entry["field_types"].setdefault(key, set()).add(kind)

            # Discover nested collection names.
            for sub in snapshot.reference.collections():
                entry["subcollections"].add(sub.id)

    except Exception as exc:
        entry["error"] = type(exc).__name__

    entry["field_names"] = sorted(entry["field_names"])
    entry["subcollections"] = sorted(entry["subcollections"])

    entry["field_types"] = {
        key: sorted(values)
        for key, values in entry["field_types"].items()
    }

    report["schema"].append(entry)

    # Inspect nested collections without exposing document IDs.
    for sub_name in entry["subcollections"]:
        for doc_ref in collection.list_documents():
            sub = doc_ref.collection(sub_name)
            inspect_collection(sub, path + "/{document}")

try:
    for collection in db.collections():
        inspect_collection(collection)

    with open(
        "firestore_schema_audit.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\nSaved firestore_schema_audit.json")

except Exception as exc:
    print("Audit failed:", type(exc).__name__)
