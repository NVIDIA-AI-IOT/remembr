"""Integration tests for MilvusMemory lifelong memory management.

Backend selection (server / Milvus Lite / skip) lives in milvus_test_utils
and the shared `milvus_db_address` fixture. The HuggingFace embedder is
replaced with a lightweight deterministic stand-in so the tests do not
download a model.
"""

import pytest

pymilvus = pytest.importorskip('pymilvus')
pytest.importorskip('langchain_community')
pytest.importorskip('langchain_huggingface')

from milvus_test_utils import (
    MILVUS_PORT,
    FakeEmbedder,
    milvus_lite_available,
    milvus_server_reachable,
)

COLLECTION = 'test_lifelong_memory_policy'

T0 = 1_722_000_000.0

pytestmark = pytest.mark.skipif(
    not (milvus_server_reachable() or milvus_lite_available()),
    reason='no MilvusDB server reachable and milvus-lite is not installed')


@pytest.fixture
def make_memory(monkeypatch, milvus_db_address):
    import remembr.memory.milvus_memory as milvus_memory_module

    monkeypatch.setattr(milvus_memory_module, 'HuggingFaceEmbeddings',
                        lambda model_name: FakeEmbedder())

    from remembr.memory.milvus_memory import MilvusMemory

    created = []

    def factory(**kwargs):
        mem = MilvusMemory(COLLECTION, db_ip=milvus_db_address, db_port=MILVUS_PORT, **kwargs)
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


def test_camera_id_roundtrip(memory):
    from remembr.memory.memory import MemoryItem

    memory.insert(MemoryItem(caption='a plant to the left', time=T0,
                             position=[0.0, 0.0, 0.0], theta=1.57, camera_id='left'))
    _insert(memory, 'i see a desk', T0 + 5)  # no camera_id -> stored as ''
    memory.milv_wrapper.collection.flush()

    by_caption = {r.item.caption: r.item for r in memory.get_all()}
    assert by_caption['a plant to the left'].camera_id == 'left'
    assert by_caption['i see a desk'].camera_id == ''


def test_insert_into_pre_camera_collection(monkeypatch, milvus_db_address):
    """Collections created before the camera_id field keep working."""

    from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                          connections, utility)

    collection_name = 'test_pre_camera_collection'

    if milvus_db_address.endswith('.db'):
        connections.connect(uri=milvus_db_address)
    else:
        connections.connect(host=milvus_db_address, port=MILVUS_PORT)
    utility.drop_collection(collection_name)
    legacy_fields = [
        FieldSchema(name='id', dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=1000),
        FieldSchema(name='text_embedding', dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name='position', dtype=DataType.FLOAT_VECTOR, dim=3),
        FieldSchema(name='theta', dtype=DataType.FLOAT),
        FieldSchema(name='time', dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema(name='caption', dtype=DataType.VARCHAR, max_length=3000),
    ]
    Collection(name=collection_name, schema=CollectionSchema(fields=legacy_fields))

    import remembr.memory.milvus_memory as milvus_memory_module
    monkeypatch.setattr(milvus_memory_module, 'HuggingFaceEmbeddings',
                        lambda model_name: FakeEmbedder())
    from remembr.memory.memory import MemoryItem
    from remembr.memory.milvus_memory import MilvusMemory

    memory = MilvusMemory(collection_name, db_ip=milvus_db_address, db_port=MILVUS_PORT)
    try:
        assert not memory.has_camera_field
        # camera_id is silently dropped instead of failing the insert
        memory.insert(MemoryItem(caption='i see a desk', time=T0,
                                 position=[0.0, 0.0, 0.0], theta=0.0, camera_id='front'))
        memory.milv_wrapper.collection.flush()

        records = memory.get_all()
        assert len(records) == 1
        assert records[0].item.caption == 'i see a desk'
        assert records[0].item.camera_id == ''
    finally:
        memory.milv_wrapper.drop_collection()


def test_get_all_on_empty_collection(memory):
    # Regression: vector output fields on an empty collection crash
    # milvus-lite; get_all must probe first and return [].
    assert memory.get_all() == []


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


def test_get_all_without_query_iterator_fallback(memory):
    # Exercise the fallback branch for old pymilvus versions by hiding
    # query_iterator behind a proxy.
    _insert(memory, 'i see a desk', T0)
    _insert(memory, 'i see a hallway', T0 + 5)
    memory.milv_wrapper.collection.flush()

    class NoIteratorProxy:
        def __init__(self, collection):
            self._collection = collection

        def __getattr__(self, name):
            if name == 'query_iterator':
                raise AttributeError(name)
            return getattr(self._collection, name)

    real_collection = memory.milv_wrapper.collection
    memory.milv_wrapper.collection = NoIteratorProxy(real_collection)
    try:
        records = memory.get_all()
    finally:
        memory.milv_wrapper.collection = real_collection

    assert sorted(r.item.caption for r in records) == ['i see a desk', 'i see a hallway']


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
