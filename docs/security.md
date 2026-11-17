# Security

## Access

`firestore.rules` denies all client SDK access. The Admin SDK bypasses those rules, so ownership is enforced in `DataPlane`.

- Patient methods take the authenticated internal user id and reject a different owner.
- Clerk mapping is created inside a transaction and will not attach a second internal id to an existing Clerk id.
- Document ids reject path separators and empty values.
- `system_config` rejects secret-like keys.
- Private identity is encrypted with `IDENTITY_ENCRYPTION_KEY` through Fernet. The key is not written to Firestore. Set `IDENTITY_ENCRYPTION_REQUIRED=true` in production after the key is in Secret Manager or the VM environment file.
- Audit metadata drops name and secret-like keys.
- Logs for mirror failures record the exception type and the patient id, not the identity payload.
- `/ready` does not return secrets.

Admin deletion is still limited to Clerk ids in `ADMIN_CLERK_USER_IDS`. It now also deletes the nested patient tree for that owner. Disk files and RL logs are removed only when no other account owns the same patient id, which is the existing rule.

## Rate limiting and errors

Chart, chat, read, and unlock routes keep the existing process-local rate limiter. Error responses from `app_core.http` stay generic. A forged id or cross-user read raises `ForgedIdentityError` or `OwnershipError` instead of returning another patient's document.

## Retention

`app.services.delete_patient` deletes the nested patient documents for one owner. It does not delete shared knowledge, RL config, or another account's copy. Cloud Storage model artifacts, when registered on `rl_models`, have to be deleted from the bucket named by `artifact_uri`. The patient folder on disk is still removed by the admin route when the patient has no other owner.

## Production switch

Do not set `KB_STRICT=true` until `python -m migrations.migrate_firestore` has been applied against the intended database. Strict mode refuses chart calculation when the knowledge base is not the Firestore source, including during an outage. That is intentional. The fixture source is only for tests and offline development, and `/ready` reports it.
