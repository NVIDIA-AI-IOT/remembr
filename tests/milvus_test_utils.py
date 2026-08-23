"""Shared helpers for the Milvus-backed tests.

The integration tests run against a MilvusDB server on 127.0.0.1:19530 when
one is reachable (see README setup), and otherwise fall back to Milvus Lite
(an embedded, file-backed Milvus, installed separately via
`pip install milvus-lite`) so they need no docker.
"""

import hashlib
import socket

MILVUS_IP = '127.0.0.1'
MILVUS_PORT = 19530


def milvus_server_reachable() -> bool:
    try:
        with socket.create_connection((MILVUS_IP, MILVUS_PORT), timeout=2):
            return True
    except OSError:
        return False


def milvus_lite_available() -> bool:
    try:
        import milvus_lite  # noqa: F401
        return True
    except ImportError:
        return False


class FakeEmbedder:
    """Deterministic bag-of-words embedding: same words -> same vector.

    Stands in for the HuggingFace embedder so tests do not download a model.
    """

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
