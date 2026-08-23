"""Integration tests for MilvusMemory lifelong memory management.

Runs against a MilvusDB server on 127.0.0.1:19530 when one is reachable (see
README setup), and otherwise falls back to Milvus Lite (an embedded,
file-backed Milvus, installed separately via `pip install milvus-lite`) so the
tests need no docker. The module is skipped when neither backend is available. The HuggingFace embedder
is replaced with a lightweight deterministic stand-in so the tests do not
download a model.
"""

import hashlib
import socket

import pytest

pymilvus = pytest.importorskip('pymilvus')
pytest.importorskip('langchain_community')
pytest.importorskip('langchain_huggingface')

MILVUS_IP = '127.0.0.1'
MILVUS_PORT = 19530
COLLECTION = 'test_lifelong_memory_policy'

T0 = 1_722_000_000.0


def _milvus_server_reachable() -> bool:
    try:
        with socket.create_connection((MILVUS_IP, MILVUS_PORT), timeout=2):
            return True
    except OSError:
        return False


def _milvus_lite_available() -> bool:
    try:
        import milvus_lite  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_SERVER = _milvus_server_reachable()
_HAS_LITE = _milvus_lite_available()

pytestmark = pytest.mark.skipif(
    not (_HAS_SERVER or _HAS_LITE),
    reason=f'no MilvusDB at {MILVUS_IP}:{MILVUS_PORT} and milvus-lite is not installed')


class FakeEmbedder:
    """Deterministic bag-of-words embedding: same words -> same vector."""

    DIM = 1024

    def embed_query(self, text: str):
        vector = [0.0] * self.DIM
        for token in (text or '').lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], 'little') % self.DIM
            vector[index] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


@pytest.fixture(scope='session')
def db_address(tmp_path_factory):
    if _HAS_SERVER:
        return MILVUS_IP
    # pymilvus keeps one connection per alias, so every test must share the
    # same Milvus Lite file; collections are dropped between tests instead.
    return str(tmp_path_factory.mktemp('milvus') / 'remembr_test.db')


@pytest.fixture
def make_memory(monkeypatch, db_address):
    import remembr.memory.milvus_memory as milvus_memory_module

    monkeypatch.setattr(milvus_memory_module, 'HuggingFaceEmbeddings',
                        lambda model_name: FakeEmbedder())

    from remembr.memory.milvus_memory import MilvusMemory

    created = []

    def factory(**kwargs):
        mem = MilvusMemory(COLLECTION, db_ip=db_address, db_port=MILVUS_PORT, **kwargs)
        mem.reset(drop_collection=True)
        created.append(mem)
        return mem

    yield factory

    for mem in created:
        mem.milv_wrapper.drop_collection()


@pytest.fixture
def memory(make_memory):
    return make_memory()


def _insert(memory, caption, t, position=(0.0, 0.0, 0.0)):
    from remembr.memory.memory import MemoryItem

    memory.insert(MemoryItem(caption=caption, time=t, position=list(position), theta=0.0))


def test_get_all_roundtrip(memory):
    _insert(memory, 'i see a desk', T0, position=(1.0, 2.0, 3.0))
    _insert(memory, 'i see a hallway', T0 + 5, position=(4.0, 5.0, 6.0))
    memory.milv_wrapper.collection.flush()

    records = memory.get_all()
    assert len(records) == 2

    by_caption = {r.item.caption: r for r in records}
    assert set(by_caption) == {'i see a desk', 'i see a hallway'}
    desk = by_caption['i see a desk']
    assert desk.item.position == pytest.approx([1.0, 2.0, 3.0])
    assert desk.item.time == pytest.approx(T0, abs=1.0)
    assert len(desk.embedding) == FakeEmbedder.DIM


def test_remove_deletes_by_id(memory):
    _insert(memory, 'i see a desk', T0)
    _insert(memory, 'i see a hallway', T0 + 5)
    memory.milv_wrapper.collection.flush()

    records = memory.get_all()
    target = next(r for r in records if r.item.caption == 'i see a desk')
    memory.remove([target.id])

    remaining = memory.get_all()
    assert [r.item.caption for r in remaining] == ['i see a hallway']


def test_apply_policy_prunes_stale_duplicates(memory):
    from remembr.memory.memory_policy import StaleDuplicatePolicy

    for i in range(5):
        _insert(memory, 'i see a desk', T0 + i, position=(0.1 * i, 0.0, 0.0))
    _insert(memory, 'a person walks past holding boxes', T0 + 2, position=(0.2, 0.0, 0.0))
    _insert(memory, 'i see a desk', T0 + 10_000, position=(0.2, 0.0, 0.0))
    memory.milv_wrapper.collection.flush()

    policy = StaleDuplicatePolicy(max_age=600.0, position_radius=1.0)
    result = memory.apply_policy(policy)

    assert result.num_scanned == 7
    assert result.num_dropped == 5

    remaining = memory.get_all()
    captions = sorted(r.item.caption for r in remaining)
    assert captions == ['a person walks past holding boxes', 'i see a desk']


def test_search_paths_still_work(memory):
    # Guards the connection reuse between MilvusWrapper and the langchain
    # vectorstores (notably for the Milvus Lite backend).
    _insert(memory, 'i see a desk with a laptop', T0, position=(1.0, 0.0, 0.0))
    _insert(memory, 'i see a hallway with doors', T0 + 5, position=(10.0, 0.0, 0.0))
    memory.milv_wrapper.collection.flush()

    text_result = memory.search_by_text('desk laptop')
    assert 'desk' in text_result

    pos_result = memory.search_by_position((10.0, 0.0, 0.0))
    assert 'hallway' in pos_result


def test_auto_prune_on_insert(make_memory):
    from remembr.memory.memory_policy import StaleDuplicatePolicy

    policy = StaleDuplicatePolicy(max_age=600.0, position_radius=1.0)
    memory = make_memory(policy=policy, prune_every=10)

    for i in range(9):
        _insert(memory, 'i see a desk', T0 + i)
    _insert(memory, 'i see a desk', T0 + 10_000)  # 10th insert triggers pruning
    memory.milv_wrapper.collection.flush()

    remaining = memory.get_all()
    assert len(remaining) == 1
    assert remaining[0].item.time == pytest.approx(T0 + 10_000, abs=1.0)
