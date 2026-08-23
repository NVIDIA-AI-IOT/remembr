import pytest

from milvus_test_utils import (
    MILVUS_IP,
    milvus_lite_available,
    milvus_server_reachable,
)


@pytest.fixture(scope='session')
def milvus_db_address(tmp_path_factory):
    """Address for MilvusMemory: a running server if reachable, else a
    Milvus Lite file shared by the whole session (pymilvus keeps one
    connection per alias, so every test must use the same Lite file;
    collections are dropped between tests instead)."""
    if milvus_server_reachable():
        return MILVUS_IP
    if not milvus_lite_available():
        pytest.skip('no MilvusDB server reachable and milvus-lite is not installed')
    return str(tmp_path_factory.mktemp('milvus') / 'remembr_test.db')
