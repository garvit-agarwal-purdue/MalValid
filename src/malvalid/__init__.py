"""MalValid: a pre-deployment reliability & security gate for malware-detection models."""

__version__ = "0.1.0"

REPORT_SCHEMA_VERSION = "malvalid-report/1"
CORPUS_FORMAT = "malvalid-corpus/1"

# The project was called "malguard" up to 0.1.0a1. Files written under that name are still read:
# corpora (the canonical EMBER builds are ~10 GB and their pinned content hashes must keep matching),
# reports, run-state files and model specs. These identifiers are only ever read, never written.
LEGACY_CORPUS_FORMATS: tuple[str, ...] = ("malguard-corpus/1",)
LEGACY_REPORT_SCHEMA_VERSIONS: tuple[str, ...] = ("malguard-report/1",)
#: Every corpus format this version can load.
SUPPORTED_CORPUS_FORMATS: tuple[str, ...] = (CORPUS_FORMAT, *LEGACY_CORPUS_FORMATS)
#: Every report schema version this version renders without a "different schema" warning.
SUPPORTED_REPORT_SCHEMA_VERSIONS: tuple[str, ...] = (REPORT_SCHEMA_VERSION, *LEGACY_REPORT_SCHEMA_VERSIONS)
