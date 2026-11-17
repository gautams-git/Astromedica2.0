
import json

with open("final_data_audit.json", encoding="utf-8") as f:
    data = json.load(f)

summary = {
    "project": data.get("project"),
    "total_collections": len(data.get("collections", [])),
    "collections": []
}

for item in data.get("collections", []):
    fields = item.get("fields", {})

    summary["collections"].append({
        "path": item.get("path"),
        "sampled_documents": item.get("sampled_documents"),
        "field_count": len(fields),
        "field_names": sorted(fields.keys()),
        "subcollections": item.get("subcollections", []),
        "error": item.get("error")
    })

with open("final_data_compact.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Compact audit created.")
print("Collections:", summary["total_collections"])
print("File: final_data_compact.json")
