
import os
import json
from pathlib import Path

ROOT = Path.cwd()

IGNORE = {
    ".git", "node_modules", ".venv", "venv",
    "__pycache__", ".next", "dist", "build",
    ".dart_tool", ".idea", ".vscode"
}

SENSITIVE = {
    ".env", ".env.local", ".env.production",
    "credentials.json", "service-account.json",
    "id_rsa", "id_ed25519"
}

results = {
    "workspace": str(ROOT.name),
    "files": [],
    "config_files": [],
    "potential_data_files": [],
    "directories": [],
}

for current, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in IGNORE]

    relative = Path(current).relative_to(ROOT)

    if str(relative) != ".":
        results["directories"].append(str(relative))

    for filename in files:
        path = Path(current) / filename
        rel = str(path.relative_to(ROOT))
        suffix = path.suffix.lower()

        if filename.lower() in SENSITIVE:
            continue

        results["files"].append(rel)

        if filename.lower() in {
            "package.json", "requirements.txt",
            "pyproject.toml", "firebase.json",
            "firestore.rules", "firestore.indexes.json",
            "dockerfile", "cloudbuild.yaml",
            "terraform.tf", "pubspec.yaml",
            "go.mod", "pom.xml"
        }:
            results["config_files"].append(rel)

        if suffix in {
            ".csv", ".json", ".pdf", ".docx",
            ".xlsx", ".txt", ".md", ".pkl",
            ".faiss", ".sqlite", ".db"
        }:
            results["potential_data_files"].append(rel)

print(json.dumps(results, indent=2))
