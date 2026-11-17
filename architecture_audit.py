
import ast
import json
from pathlib import Path

ROOT = Path.cwd()

IGNORE = {
    ".git", "venv", ".venv", "__pycache__",
    "node_modules", "dist", "build"
}

report = {
    "project": ROOT.name,
    "python_files": [],
    "firestore_references": {},
    "google_cloud_references": {},
    "errors": []
}

for path in ROOT.rglob("*.py"):
    if any(part in IGNORE for part in path.parts):
        continue

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        functions = []
        classes = []
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

            elif isinstance(node, ast.Import):
                imports.update(x.name for x in node.names)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

        report["python_files"].append({
            "file": str(path.relative_to(ROOT)),
            "functions": sorted(set(functions)),
            "classes": sorted(set(classes)),
            "imports": sorted(imports)
        })

        # Extract collection names without executing code.
        import re

        collections = set(
            re.findall(
                r"\.collection\(\s*['\"]([^'\"]+)['\"]\s*\)",
                source
            )
        )

        if collections:
            report["firestore_references"][
                str(path.relative_to(ROOT))
            ] = sorted(collections)

        cloud_terms = [
            "firestore", "storage", "firebase",
            "bigquery", "secretmanager",
            "cloudtasks", "pubsub"
        ]

        found = [
            term for term in cloud_terms
            if term in source.lower()
        ]

        if found:
            report["google_cloud_references"][
                str(path.relative_to(ROOT))
            ] = found

    except Exception as exc:
        report["errors"].append({
            "file": str(path.relative_to(ROOT)),
            "error": type(exc).__name__
        })

with open("architecture_audit.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("Architecture audit completed.")
print("Python files:", len(report["python_files"]))
print("Firestore references:", len(report["firestore_references"]))
print("Google Cloud references:", len(report["google_cloud_references"]))
print("Saved: architecture_audit.json")
