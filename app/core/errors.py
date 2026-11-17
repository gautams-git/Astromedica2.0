"""Explicit data-plane errors. Callers must not treat these as empty results."""


class DataPlaneError(Exception):
    """Base error for durable storage."""


class FirestoreUnavailableError(DataPlaneError):
    """The configured Firestore backend could not be reached."""


class MissingDatasetError(DataPlaneError):
    """A required knowledge-base dataset is not in the configured source."""


class DatasetDriftError(DataPlaneError):
    """Target dataset exists but its checksum does not match the source."""


class OwnershipError(DataPlaneError):
    """The authenticated user does not own the requested patient record."""


class ForgedIdentityError(DataPlaneError):
    """A caller-supplied identifier is not a verified account identity."""


class DocumentTooLargeError(DataPlaneError):
    """A value cannot be stored without dropping data, which is not allowed."""


class MigrationRefusedError(DataPlaneError):
    """A migration write was refused by a safety guard."""
