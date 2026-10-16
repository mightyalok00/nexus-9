"""Evidence Ledger: Tamper-evident ledger recording experiments and evaluations."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from nexus9.config import PROJECT_ROOT, settings

class EvidenceLedger:
    """Append-only verifiable evidence ledger for reproducible research."""

    def __init__(self, ledger_file: Optional[Path | str] = None):
        self.ledger_file = Path(ledger_file or (PROJECT_ROOT / settings.paths.evidence_ledger))
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.ledger_file.exists():
            self._save_entries([])

    def _load_entries(self) -> List[Dict[str, Any]]:
        try:
            with open(self.ledger_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_entries(self, entries: List[Dict[str, Any]]) -> None:
        with open(self.ledger_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    def record_entry(
        self,
        experiment_type: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        agent_notes: str = "",
    ) -> Dict[str, Any]:
        """Append a cryptographically hashed evidence entry to the ledger."""
        entries = self._load_entries()
        prev_hash = entries[-1]["hash"] if entries else "GENESIS_NEXUS9"

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload = {
            "index": len(entries),
            "timestamp": timestamp,
            "experiment_type": experiment_type,
            "parameters": parameters,
            "metrics": metrics,
            "agent_notes": agent_notes,
            "prev_hash": prev_hash,
        }

        # SHA-256 integrity hash
        payload_str = json.dumps(payload, sort_keys=True)
        entry_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        payload["hash"] = entry_hash

        entries.append(payload)
        self._save_entries(entries)
        return payload

    def get_all_entries(self) -> List[Dict[str, Any]]:
        return self._load_entries()
