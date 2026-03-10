"""
Section Checkpoint Module for Notebook Resume Capability

Saves/loads arbitrary state dicts as pickle files so that long-running
notebooks can resume from the last completed section after a disconnect.

Checkpoints are stored in results/{dataset_id}/checkpoints/.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SectionCheckpoint:
    """Checkpoint manager for notebook sections.

    Parameters
    ----------
    dataset_identifier : str
        Dataset identifier (e.g. ``"breast-cancer-data"``).
    base_dir : str
        Root results directory (default ``"results"``).
    """

    def __init__(self, dataset_identifier: str, base_dir: str = "results"):
        self.dataset_identifier = dataset_identifier
        self.checkpoint_dir = Path(base_dir) / dataset_identifier / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, section_id: str) -> Path:
        safe_name = section_id.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"{safe_name}.pkl"

    def save(self, section_id: str, state: dict) -> Path:
        """Persist *state* dict for *section_id*."""
        path = self._path(section_id)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[CHECKPOINT] Saved {section_id} -> {path}")
        return path

    def load(self, section_id: str) -> Optional[dict]:
        """Load state dict for *section_id*, or ``None`` if missing."""
        path = self._path(section_id)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            state = pickle.load(f)
        logger.info(f"[CHECKPOINT] Loaded {section_id} <- {path}")
        return state

    def exists(self, section_id: str) -> bool:
        return self._path(section_id).exists()

    def clear(self, section_id: str = None) -> int:
        """Remove checkpoint(s). If *section_id* is ``None``, remove all."""
        removed = 0
        if section_id is not None:
            path = self._path(section_id)
            if path.exists():
                path.unlink()
                removed = 1
        else:
            for path in self.checkpoint_dir.glob("*.pkl"):
                path.unlink()
                removed += 1
        return removed

    def clear_all(self) -> int:
        """Remove all checkpoints AND flush Section 4 optimization studies."""
        removed = self.clear()
        try:
            from src.models.staged_optimization import flush_previous_runs
            flush_previous_runs(self.dataset_identifier)
        except Exception:
            pass
        return removed

    def list_checkpoints(self) -> List[str]:
        """Return sorted list of existing checkpoint section IDs."""
        return sorted(p.stem for p in self.checkpoint_dir.glob("*.pkl"))
