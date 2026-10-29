"""Shared Part A paths and provenance helpers."""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "part_a"
RAW = PROJECT / "data" / "raw" / "face_segment"
MANIFEST = PROJECT / "data" / "split_manifest.csv"

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def content_fingerprint(rows):
    """Bind reviews to sample content, independently of its split assignment."""
    payload = sorted((r["sample_id"], r["image_sha256"], r["label_sha256"]) for r in rows)
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
