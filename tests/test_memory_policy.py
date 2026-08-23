import random

import pytest

from remembr.memory.memory import MemoryItem
from remembr.memory.memory_policy import (
    MemoryRecord,
    StaleDuplicatePolicy,
    _cosine_similarity,
    _jaccard_similarity,
)

T0 = 1_722_000_000.0  # arbitrary "start of run" epoch seconds


def rec(id, caption, t, position=(0.0, 0.0, 0.0), embedding=None):
    item = MemoryItem(caption=caption, time=t, position=list(position), theta=0.0)
    return MemoryRecord(id=id, item=item, embedding=embedding)


def default_policy(**kwargs):
    params = dict(max_age=600.0, position_radius=1.0)
    params.update(kwargs)
    return StaleDuplicatePolicy(**params)


class TestSelection:

    def test_empty_records(self):
        result = default_policy().select_for_removal([])
        assert result.drop_ids == []
        assert result.num_scanned == 0
        assert result.num_dropped == 0

    def test_stale_duplicate_dropped_keeps_newest(self):
        records = [
            rec('old', 'i see a desk', T0),
            rec('new', 'i see a desk', T0 + 10_000),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == ['old']
        assert result.num_scanned == 2

    def test_fresh_duplicates_are_kept(self):
        # Both entries are within max_age of the newest entry.
        records = [
            rec('a', 'i see a desk', T0),
            rec('b', 'i see a desk', T0 + 30),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == []

    def test_stale_unique_caption_is_kept(self):
        records = [
            rec('old', 'i see a fire extinguisher', T0),
            rec('new', 'i see a desk', T0 + 10_000),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == []

    def test_distant_duplicates_are_kept(self):
        records = [
            rec('old', 'i see a desk', T0, position=(50.0, 0.0, 0.0)),
            rec('new', 'i see a desk', T0 + 10_000, position=(0.0, 0.0, 0.0)),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == []

    def test_cluster_consolidates_to_newest(self):
        # Robot parked at a desk for a while: many stale near-duplicates plus
        # one fresh one. Everything but the newest should go.
        records = [rec(f'dup{i}', 'i see a desk and a chair', T0 + i,
                       position=(0.1 * i, 0.0, 0.0)) for i in range(5)]
        records.append(rec('fresh', 'i see a desk and a chair', T0 + 10_000, position=(0.2, 0.0, 0.0)))
        result = default_policy().select_for_removal(records)
        assert sorted(result.drop_ids) == ['dup0', 'dup1', 'dup2', 'dup3', 'dup4']

    def test_chained_duplicates_only_compare_against_kept(self):
        # a-b are within radius, b-c are within radius, but a-c are not.
        # Scanning newest-first keeps c, drops b (near c), then a must be
        # compared against c (kept), not b (dropped) - so a survives.
        records = [
            rec('a', 'i see a desk', T0, position=(0.0, 0.0, 0.0)),
            rec('b', 'i see a desk', T0 + 1, position=(0.9, 0.0, 0.0)),
            rec('c', 'i see a desk', T0 + 10_000, position=(1.8, 0.0, 0.0)),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == ['b']

    def test_boundary_distance_counts_as_nearby(self):
        records = [
            rec('old', 'i see a desk', T0, position=(1.0, 0.0, 0.0)),
            rec('new', 'i see a desk', T0 + 10_000, position=(0.0, 0.0, 0.0)),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == ['old']

    def test_nearby_across_grid_cells(self):
        # Two points in different grid cells (floor(1.05)=1 vs floor(0.9)=0)
        # but still within the radius, so the 3x3x3 neighbor search must find
        # the kept entry.
        records = [
            rec('old', 'i see a desk', T0, position=(1.05, 0.0, 0.0)),
            rec('new', 'i see a desk', T0 + 10_000, position=(0.9, 0.0, 0.0)),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == ['old']

    def test_negative_coordinates(self):
        records = [
            rec('old', 'i see a desk', T0, position=(-0.4, -0.4, 0.0)),
            rec('new', 'i see a desk', T0 + 10_000, position=(0.3, 0.3, 0.0)),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == ['old']

    def test_empty_captions_count_as_duplicates(self):
        records = [
            rec('old', '', T0),
            rec('new', '', T0 + 10_000),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == ['old']

    def test_order_independence(self):
        base = [rec(f'dup{i}', 'the hallway is empty', T0 + i) for i in range(10)]
        base.append(rec('fresh', 'the hallway is empty', T0 + 10_000))
        base.append(rec('unique', 'a person walks by holding boxes', T0 + 5, position=(0.5, 0.0, 0.0)))

        expected = {f'dup{i}' for i in range(10)}
        for seed in range(3):
            shuffled = base[:]
            random.Random(seed).shuffle(shuffled)
            result = default_policy().select_for_removal(shuffled)
            assert set(result.drop_ids) == expected


class TestNowHandling:

    def test_now_defaults_to_newest_entry(self):
        # All entries ancient in wall-clock terms, but the newest defines "now",
        # so a replayed log is not wiped out wholesale.
        records = [
            rec('a', 'i see a desk', 100.0),
            rec('b', 'i see a desk', 150.0),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == []

    def test_explicit_now_makes_entries_stale(self):
        records = [
            rec('a', 'i see a desk', 100.0),
            rec('b', 'i see a desk', 150.0),
        ]
        result = default_policy().select_for_removal(records, now=100_000.0)
        assert result.drop_ids == ['a']


class TestSimilarityModes:

    def test_similar_embeddings_dropped(self):
        emb_a = [1.0, 0.0, 0.01]
        emb_b = [1.0, 0.01, 0.0]
        records = [
            rec('old', 'caption one', T0, embedding=emb_a),
            rec('new', 'caption two', T0 + 10_000, embedding=emb_b),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == ['old']

    def test_dissimilar_embeddings_kept_despite_same_caption(self):
        # When embeddings are available they take precedence over raw text.
        records = [
            rec('old', 'i see a desk', T0, embedding=[1.0, 0.0, 0.0]),
            rec('new', 'i see a desk', T0 + 10_000, embedding=[0.0, 1.0, 0.0]),
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == []

    def test_missing_embedding_falls_back_to_text(self):
        records = [
            rec('old', 'i see a desk', T0, embedding=[1.0, 0.0, 0.0]),
            rec('new', 'i see a desk', T0 + 10_000),  # no embedding
        ]
        result = default_policy().select_for_removal(records)
        assert result.drop_ids == ['old']

    def test_lazy_embedding_loader_only_called_for_compared_records(self):
        loads = []

        def loader_for(record_id, vector):
            def load():
                loads.append(record_id)
                return vector
            return load

        emb = [1.0, 0.0, 0.0]
        records = [
            rec('old', 'i see a desk', T0),
            rec('new', 'i see a desk', T0 + 10_000),
            rec('far', 'i see a desk', T0, position=(100.0, 0.0, 0.0)),
        ]
        records[0].embedding_loader = loader_for('old', emb)
        records[1].embedding_loader = loader_for('new', emb)
        records[2].embedding_loader = loader_for('far', emb)

        result = default_policy().select_for_removal(records)

        # 'old' duplicates 'new' nearby, so both embeddings are fetched;
        # 'far' is never compared against anything, so its loader never runs.
        assert result.drop_ids == ['old']
        assert set(loads) == {'old', 'new'}

    def test_get_embedding_caches_loader_result(self):
        from remembr.memory.memory_policy import MemoryRecord
        calls = []

        def load():
            calls.append(1)
            return [1.0, 2.0]

        record = rec('a', 'i see a desk', T0)
        record.embedding_loader = load
        assert record.get_embedding() == [1.0, 2.0]
        assert record.get_embedding() == [1.0, 2.0]
        assert len(calls) == 1

    def test_text_similarity_threshold_respected(self):
        strict = default_policy(text_similarity_threshold=1.0)
        records = [
            rec('old', 'i see a desk and a chair', T0),
            rec('new', 'i see a desk and a lamp', T0 + 10_000),
        ]
        assert strict.select_for_removal(records).drop_ids == []

        loose = default_policy(text_similarity_threshold=0.5)
        assert loose.select_for_removal(records).drop_ids == ['old']


class TestValidation:

    def test_negative_max_age_rejected(self):
        with pytest.raises(ValueError):
            StaleDuplicatePolicy(max_age=-1.0)

    def test_nonpositive_radius_rejected(self):
        with pytest.raises(ValueError):
            StaleDuplicatePolicy(position_radius=0.0)


class TestSimilarityHelpers:

    def test_cosine_identical(self):
        assert _cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_jaccard_identical(self):
        assert _jaccard_similarity('i see a desk', 'i see a desk') == pytest.approx(1.0)

    def test_jaccard_case_insensitive(self):
        assert _jaccard_similarity('I SEE A DESK', 'i see a desk') == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        assert _jaccard_similarity('red apple', 'blue chair') == 0.0

    def test_jaccard_empty_vs_nonempty(self):
        assert _jaccard_similarity('', 'i see a desk') == 0.0

    def test_jaccard_both_empty(self):
        assert _jaccard_similarity('', '') == 1.0
