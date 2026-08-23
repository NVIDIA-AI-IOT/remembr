import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from remembr.memory.memory import MemoryItem


@dataclass
class MemoryRecord:
    """A stored memory entry as seen by a management policy.

    Wraps a MemoryItem together with its database id and (optionally) the
    caption embedding that the backing store keeps for it. Backends can set
    ``embedding_loader`` instead of ``embedding`` to fetch the vector lazily:
    policies only compare a small subset of entries, so this avoids pulling
    every stored embedding out of the database on each pruning pass.
    """
    id: str
    item: MemoryItem
    embedding: Optional[List[float]] = None
    embedding_loader: Optional[Callable[[], Optional[List[float]]]] = None

    def get_embedding(self) -> Optional[List[float]]:
        if self.embedding is None and self.embedding_loader is not None:
            self.embedding = self.embedding_loader()
            self.embedding_loader = None
        return self.embedding


@dataclass
class PolicyResult:
    """Outcome of applying a memory management policy."""
    drop_ids: List[str] = field(default_factory=list)
    num_scanned: int = 0

    @property
    def num_dropped(self) -> int:
        return len(self.drop_ids)


class MemoryPolicy:
    """Interface for lifelong memory management policies.

    A policy inspects the full set of stored memories and decides which
    entries should be removed (e.g. because they are stale or redundant).
    Policies only *select* entries; the Memory backend performs the actual
    deletion via ``Memory.apply_policy``.
    """

    def select_for_removal(self, records: List[MemoryRecord], now: float = None) -> PolicyResult:
        raise NotImplementedError


class StaleDuplicatePolicy(MemoryPolicy):
    """Drop sufficiently old entries that duplicate a nearby, newer observation.

    This is an intentionally simple lifelong-memory policy: as a robot keeps
    running, its memory fills up with near-identical captions recorded at the
    same place (e.g. "I see a desk" every few seconds while docked). This
    policy walks the memory from newest to oldest and drops an entry only if
    all of the following hold:

    1. The entry is stale: older than ``max_age`` seconds relative to ``now``
       (by default, ``now`` is the timestamp of the newest entry, which works
       for both live robots and replayed logs).
    2. A kept (newer or same-age) entry lies within ``position_radius`` meters.
    3. That nearby kept entry has a sufficiently similar caption:
       cosine similarity of the caption embeddings >=
       ``embedding_similarity_threshold`` when both embeddings are available,
       otherwise word-level Jaccard similarity >= ``text_similarity_threshold``.

    The newest observation of each near-duplicate cluster therefore survives,
    consolidating redundant history down to one representative entry while
    never touching recent memories or unique ones.
    """

    def __init__(self, max_age: float = 3600.0, position_radius: float = 1.0,
                 embedding_similarity_threshold: float = 0.9,
                 text_similarity_threshold: float = 0.7):
        if max_age < 0:
            raise ValueError("max_age must be >= 0")
        if position_radius <= 0:
            raise ValueError("position_radius must be > 0")
        self.max_age = max_age
        self.position_radius = position_radius
        self.embedding_similarity_threshold = embedding_similarity_threshold
        self.text_similarity_threshold = text_similarity_threshold

    def select_for_removal(self, records: List[MemoryRecord], now: float = None) -> PolicyResult:
        result = PolicyResult(num_scanned=len(records))
        if not records:
            return result

        if now is None:
            now = max(r.item.time for r in records)
        cutoff = now - self.max_age

        # Newest first, so the survivor of each duplicate cluster is the
        # most recent observation.
        order = sorted(range(len(records)), key=lambda i: records[i].item.time, reverse=True)

        positions = [np.asarray(r.item.position, dtype=float).reshape(-1) for r in records]

        # Coarse spatial hash: kept entries are bucketed into grid cells of
        # side position_radius, so each candidate only compares against kept
        # entries in its 3x3x3 cell neighborhood instead of the whole memory.
        kept_by_cell = {}

        for idx in order:
            rec = records[idx]
            pos = positions[idx]
            if rec.item.time < cutoff and \
                    self._has_kept_duplicate(rec, pos, records, positions, kept_by_cell):
                result.drop_ids.append(rec.id)
            else:
                kept_by_cell.setdefault(self._cell_of(pos), []).append(idx)

        return result

    def _cell_of(self, pos) -> tuple:
        return tuple(int(math.floor(c / self.position_radius)) for c in pos)

    def _neighbor_cells(self, cell):
        # Cartesian product of (c-1, c, c+1) along every axis.
        neighbors = [()]
        for c in cell:
            neighbors = [cur + (offset,) for cur in neighbors for offset in (c - 1, c, c + 1)]
        return neighbors

    def _has_kept_duplicate(self, rec, pos, records, positions, kept_by_cell) -> bool:
        for cell in self._neighbor_cells(self._cell_of(pos)):
            for kept_idx in kept_by_cell.get(cell, ()):
                if np.linalg.norm(positions[kept_idx] - pos) > self.position_radius:
                    continue
                if self._captions_similar(rec, records[kept_idx]):
                    return True
        return False

    def _captions_similar(self, a: MemoryRecord, b: MemoryRecord) -> bool:
        embedding_a = a.get_embedding()
        embedding_b = b.get_embedding()
        if embedding_a is not None and embedding_b is not None:
            return _cosine_similarity(embedding_a, embedding_b) >= self.embedding_similarity_threshold
        return _jaccard_similarity(a.item.caption, b.item.caption) >= self.text_similarity_threshold


def _cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set((text_a or '').lower().split())
    tokens_b = set((text_b or '').lower().split())
    if not tokens_a and not tokens_b:
        # Two empty captions carry the same (lack of) information.
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
