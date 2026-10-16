"""Artifact lineage tracking and Git/dataset provenance."""

import hashlib
from pathlib import Path
from typing import Dict

def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 checksum of a dataset or artifact."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_provenance_metadata(data_path: Path | str) -> Dict[str, str]:
    """Generate provenance card."""
    return {
        "file_path": str(data_path),
        "sha256": compute_file_sha256(data_path),
        "generator_version": "NEXUS-9 FutureBench v1",
    }
