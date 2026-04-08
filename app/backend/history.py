"""
LTX-2 Studio — Generation History
Simple JSON-based history store for generation results.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.backend.config import HISTORY_FILE

logger = logging.getLogger(__name__)


class HistoryManager:
    """Manages generation history with JSON persistence."""

    def __init__(self):
        self._items: list[dict[str, Any]] = []
        self._load()

    def _load(self):
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE) as f:
                    self._items = json.load(f)
            except Exception:
                self._items = []

    def _save(self):
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(self._items, f, indent=2, default=str)

    def add(
        self,
        pipeline: str,
        prompt: str,
        params: dict,
        status: str = "completed",
        output_path: str | None = None,
        duration: float | None = None,
        error: str | None = None,
    ) -> dict:
        """Add a generation record to history."""
        item = {
            "id": str(uuid.uuid4())[:8],
            "pipeline": pipeline,
            "prompt": prompt,
            "params": params,
            "status": status,
            "output_path": output_path,
            "duration": duration,
            "error": error,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "thumbnail": None,
        }
        self._items.insert(0, item)  # Newest first
        self._save()
        return item

    def list_all(self) -> list[dict]:
        return self._items

    def get(self, item_id: str) -> dict | None:
        for item in self._items:
            if item["id"] == item_id:
                return item
        return None

    def delete(self, item_id: str) -> bool:
        before = len(self._items)
        self._items = [i for i in self._items if i["id"] != item_id]
        if len(self._items) < before:
            self._save()
            return True
        return False
