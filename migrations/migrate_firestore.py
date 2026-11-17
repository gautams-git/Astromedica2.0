"""Dry-run by default. Does not delete or overwrite existing documents."""

from __future__ import annotations

import argparse
import json
import sys

from app.core.config import load_data_settings
from app.core.errors import MigrationRefusedError

PRODUCTION_PROJECT = "astrohealth-509609"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy AstroMedica data onto the target Firestore layout.")
    parser.add_argument("--apply", action="store_true", help="Write missing target documents. Default is dry-run.")
    parser.add_argument("--manifest", default="", help="Write the JSON manifest to this path.")
    parser.add_argument(
        "--allow-production-project",
        action="store_true",
        help=f"Required before --apply can touch project {PRODUCTION_PROJECT}.",
    )
    args = parser.parse_args(argv)
    settings = load_data_settings()
    if args.apply and settings.project_id == PRODUCTION_PROJECT and not args.allow_production_project:
        raise MigrationRefusedError(
            f"refusing to write {PRODUCTION_PROJECT} without --allow-production-project"
        )
    if not settings.firestore_enabled:
        print("No Firestore backend is configured. Set FIRESTORE_EMULATOR_HOST or FIRESTORE_BACKEND=firestore.")
        print("This command does not fall back to embedded files and did not write anything.")
        return 2
    from app.infrastructure.firestore.client import call_options, get_shared_client
    from app.infrastructure.firestore.firestore_store import FirestoreDocumentStore
    from app.infrastructure.firestore.repositories import DataPlane
    from migrations.plan import apply_migration, plan_migration

    client = get_shared_client(settings)
    store = FirestoreDocumentStore(client, call_options())
    if args.apply:
        report = apply_migration(DataPlane(store, encryption_key=settings.identity_encryption_key), store)
    else:
        report = plan_migration(store)
    text = json.dumps(report, indent=2, default=str)
    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    print(text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MigrationRefusedError as exc:
        print(str(exc))
        sys.exit(3)
