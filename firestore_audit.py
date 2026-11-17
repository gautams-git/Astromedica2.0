
import json
import traceback
from google.cloud import firestore

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ID = "astrohealth-509609"
MAX_DOCS_PER_COLLECTION = 300
MAX_DEPTH = 4

db = firestore.Client(project=PROJECT_ID)

report = {
    "project": PROJECT_ID,
    "audit_type": "READ_ONLY_METADATA",
    "collections": [],
    "warnings": []
}

visited = set()


# --------------------------------------------------
# COLLECTION INSPECTION
# --------------------------------------------------

def inspect_collection(collection, parent_path="", depth=0):

    collection_name = collection.id

    path = (
        f"{parent_path}/{collection_name}"
        if parent_path
        else collection_name
    )

    if path in visited or depth > MAX_DEPTH:
        return

    visited.add(path)

    item = {
        "path": path,
        "document_reference_count": 0,
        "scan_limit": MAX_DOCS_PER_COLLECTION,
        "truncated": False,
        "subcollections": []
    }

    report["collections"].append(item)

    try:
        for doc_ref in collection.list_documents():

            if item["document_reference_count"] >= MAX_DOCS_PER_COLLECTION:
                item["truncated"] = True
                break

            item["document_reference_count"] += 1

            # Discover nested subcollections
            for subcollection in doc_ref.collections():

                item["subcollections"].append(
                    subcollection.id
                )

                inspect_collection(
                    subcollection,
                    parent_path=path + "/{document}",
                    depth=depth + 1
                )

        item["subcollections"] = sorted(
            set(item["subcollections"])
        )

    except Exception as exc:

        report["warnings"].append({
            "collection": path,
            "error": type(exc).__name__
        })


# --------------------------------------------------
# MAIN AUDIT
# --------------------------------------------------

try:

    print("\nConnecting to Google Cloud Firestore...")
    print("Project:", PROJECT_ID)

    print("\nDiscovering root collections...\n")

    for collection in db.collections():

        inspect_collection(collection)

    # Save report
    output_file = "firestore_inventory.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========== FIRESTORE INVENTORY ==========\n")

    print(json.dumps(report, indent=2))

    print("\n=========================================")
    print("AUDIT COMPLETED")
    print("Output:", output_file)
    print("Collections discovered:", len(report["collections"]))
    print("Warnings:", len(report["warnings"]))

except Exception as exc:

    print("\nAUDIT FAILED:", type(exc).__name__)
    traceback.print_exc()
