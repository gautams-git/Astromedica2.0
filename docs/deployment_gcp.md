# GCP deployment

Target: Compute Engine e2-standard-2, Firestore in project `astrohealth-509609`, the existing Docker image.

## Configuration

One settings path reads the environment: `app_core.settings` for the Flask process and `app.core.config` for the data plane. Production on the VM should set:

```bash
APP_ENV=production
FIRESTORE_PROJECT_ID=astrohealth-509609
FIRESTORE_BACKEND=firestore
# After a reviewed migration:
# KB_SOURCE=firestore
# KB_STRICT=true
# RL_CONFIG_STRICT=true
CHAT_MODE=classic
CHAT_DURABLE=true
RL_HINT_MODE=off
MODEL_PROVIDER=vertex
IDENTITY_ENCRYPTION_KEY=   # from Secret Manager, not from git
```

Leave `KB_SOURCE` as `fixtures` until the seed has been applied. `/ready` will say `fixtures`. That is the explicit offline source. It is not a hidden fallback after `KB_STRICT=true`.

The container uses Application Default Credentials from the VM service account. Do not copy a service-account JSON into the image or the repo. The service account needs Firestore access and, for the existing Gemini path, Vertex AI User. It does not need a browser-facing Firebase key.

## Process

`deploy/vm-startup.sh` starts `docker-compose.prod.yml`. The compose file does not publish Firestore or Redis. Gunicorn's `graceful_timeout` is 30 seconds and `on_exit` stops the RL config refresher.

`/ready` is the container health check. It returns 200 while strict mode is off. It returns 503 when `KB_STRICT=true` and Firestore knowledge is not loaded.

Chat durability uses Firestore, so a second Gunicorn worker can restore a session after `CHAT_DURABLE=true` and `FIRESTORE_BACKEND=firestore`. Keep one worker until that has been checked on the VM. The RL queue itself is still in process memory.

## Migration

Run the dry-run from a machine authenticated to the project and read the manifest. Take a Firestore export first. Apply only with an explicit decision. The command is documented in `docs/data_migration.md`. It is not part of container startup.

## Local Docker

The same image runs locally. Point `FIRESTORE_EMULATOR_HOST` at an emulator on the Docker network if you want the Firestore code path. Do not put the emulator port on the public interface. Without the emulator, `APP_ENV=development` uses the in-memory repository and `KB_SOURCE=fixtures`.
