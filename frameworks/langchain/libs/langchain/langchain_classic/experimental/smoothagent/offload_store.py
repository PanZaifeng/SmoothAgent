"""Filesystem-backed offload store used by ``OffloadLookaheadStrategy``.

This is intentionally minimal: each call to :meth:`OffloadStore.store`
writes the full observation text to a uniquely named file under
``offload_dir`` and returns a :class:`OffloadRecord` describing the result.

The store is content-addressable per observation index — re-storing the
same logical observation produces the same file path so that repeated
``transform`` invocations are idempotent.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class OffloadRecord:
    """Metadata returned for a single offloaded observation."""

    observation_id: str
    file_path: str
    original_token_count: int
    reference_token_count: int


class OffloadStore:
    """Tiny on-disk store that persists offloaded observations."""

    def __init__(self, offload_dir: str) -> None:
        self._offload_dir = offload_dir
        self._lock = threading.Lock()
        os.makedirs(self._offload_dir, exist_ok=True)

    @property
    def offload_dir(self) -> str:
        return self._offload_dir

    def store(
        self,
        text: str,
        *,
        observation_index: int,
        original_token_count: int,
        reference_token_count: int,
    ) -> OffloadRecord:
        """Write ``text`` to disk and return its :class:`OffloadRecord`."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        observation_id = f"obs_{observation_index:06d}_{digest}"
        file_path = os.path.join(self._offload_dir, f"{observation_id}.txt")
        with self._lock:
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as handle:
                    handle.write(text)
        return OffloadRecord(
            observation_id=observation_id,
            file_path=file_path,
            original_token_count=original_token_count,
            reference_token_count=reference_token_count,
        )

    def render_reference(self, record: OffloadRecord) -> str:
        """Return the placeholder text that replaces a long observation."""
        return (
            "Observation:\n"
            f"The full observation has been written to {record.file_path}.\n"
            "Use the file-read tool if you need to inspect it again."
        )
