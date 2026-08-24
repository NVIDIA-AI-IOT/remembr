"""Runtime test of the nova_carter memory builder node's wiring.

ROS 2 itself is not required: rclpy and the message packages are replaced
with minimal stand-ins, while MilvusMemory, the pruning policy, and
common_utils.format_pose_msg run for real (against the shared Milvus
server/Lite backend). This covers what a ROS-less CI can cover: parameter
declaration, subscription wiring, the pose+caption -> MemoryItem path, and
the policy/prune_every plumbing into MilvusMemory.
"""

import importlib
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pymilvus = pytest.importorskip('pymilvus')
pytest.importorskip('langchain_community')
pytest.importorskip('langchain_huggingface')
pytest.importorskip('scipy')

from milvus_test_utils import (
    FakeEmbedder,
    milvus_lite_available,
    milvus_server_reachable,
)

NODE_DIR = str(Path(__file__).resolve().parent.parent / 'examples' / 'nova_carter_demo' / 'python')
COLLECTION = 'test_memory_builder_node'
T0 = 1_722_000_000

pytestmark = pytest.mark.skipif(
    not (milvus_server_reachable() or milvus_lite_available()),
    reason='no MilvusDB server reachable and milvus-lite is not installed')


class FakeNode:
    """Minimal stand-in for rclpy.node.Node."""

    param_overrides = {}

    def __init__(self, name):
        self.node_name = name
        self._params = {}
        self.subscriptions = []

    def declare_parameter(self, name, default=None):
        self._params[name] = self.param_overrides.get(name, default)

    def get_parameter(self, name):
        return SimpleNamespace(value=self._params[name])

    def create_subscription(self, msg_type, topic, callback, qos):
        assert callable(callback), f'subscription callback for {topic} is not callable'
        self.subscriptions.append((msg_type, topic, callback))
        return SimpleNamespace()

    def get_logger(self):
        return SimpleNamespace(info=lambda *a, **k: None,
                               warning=lambda *a, **k: None,
                               error=lambda *a, **k: None)


class PoseWithCovarianceStamped:
    pass


class String:
    def __init__(self, data=''):
        self.data = data


@pytest.fixture
def node_module(monkeypatch, milvus_db_address):
    """Import memory_builder_node with stubbed ROS modules."""

    rclpy_mod = types.ModuleType('rclpy')
    rclpy_mod.init = lambda args=None: None
    rclpy_mod.spin = lambda node: None
    rclpy_mod.shutdown = lambda: None
    rclpy_node_mod = types.ModuleType('rclpy.node')
    rclpy_node_mod.Node = FakeNode
    rclpy_mod.node = rclpy_node_mod

    geometry_mod = types.ModuleType('geometry_msgs')
    geometry_msg_mod = types.ModuleType('geometry_msgs.msg')
    geometry_msg_mod.PoseWithCovarianceStamped = PoseWithCovarianceStamped
    geometry_mod.msg = geometry_msg_mod

    std_mod = types.ModuleType('std_msgs')
    std_msg_mod = types.ModuleType('std_msgs.msg')
    std_msg_mod.String = String
    std_mod.msg = std_msg_mod

    for name, mod in [('rclpy', rclpy_mod), ('rclpy.node', rclpy_node_mod),
                      ('geometry_msgs', geometry_mod), ('geometry_msgs.msg', geometry_msg_mod),
                      ('std_msgs', std_mod), ('std_msgs.msg', std_msg_mod)]:
        monkeypatch.setitem(sys.modules, name, mod)

    import remembr.memory.milvus_memory as milvus_memory_module
    monkeypatch.setattr(milvus_memory_module, 'HuggingFaceEmbeddings',
                        lambda model_name: FakeEmbedder())

    monkeypatch.syspath_prepend(NODE_DIR)
    for cached in ('memory_builder_node', 'common_utils'):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    module = importlib.import_module('memory_builder_node')
    yield module

    for cached in ('memory_builder_node', 'common_utils'):
        sys.modules.pop(cached, None)


@pytest.fixture
def make_node(node_module, monkeypatch, milvus_db_address):
    created = []

    def factory(**param_overrides):
        params = {'db_collection': COLLECTION, 'db_ip': milvus_db_address}
        params.update(param_overrides)
        monkeypatch.setattr(FakeNode, 'param_overrides', params)
        node = node_module.MemoryBuilderNode()
        created.append(node)
        return node

    yield factory

    for node in created:
        node.memory.milv_wrapper.drop_collection()


def make_pose(x, y, t, yaw=0.0):
    msg = PoseWithCovarianceStamped()
    msg.pose = SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0,
                                    z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))))
    msg.header = SimpleNamespace(stamp=SimpleNamespace(sec=int(t), nanosec=0))
    return msg


def test_node_wiring_and_caption_insert(make_node):
    node = make_node()

    # Pruning is off by default and existing behavior is unchanged.
    assert node.memory.policy is None

    # Both subscriptions are registered with callbacks that actually exist.
    callbacks = {topic: callback for _, topic, callback in node.subscriptions}
    assert callbacks['/amcl_pose'] == node.pose_callback
    caption_callback = callbacks['/caption']
    assert callable(caption_callback)

    # A caption arriving before any pose is ignored rather than crashing.
    caption_callback(String('i see a desk'))
    assert node.memory.get_all() == []

    # pose + caption -> a MemoryItem lands in the DB with the pose's data.
    node.pose_callback(make_pose(1.0, 2.0, T0))
    caption_callback(String('i see a desk'))
    node.memory.milv_wrapper.collection.flush()

    records = node.memory.get_all()
    assert len(records) == 1
    assert records[0].item.caption == 'i see a desk'
    assert records[0].item.position == pytest.approx([1.0, 2.0, 0.0])
    assert records[0].item.time == pytest.approx(T0, abs=1.0)
    # The single-camera default tags entries as the front camera at offset 0.
    assert records[0].item.camera_id == 'front'
    assert records[0].item.theta == pytest.approx(0.0)


def test_multi_camera_captions_store_camera_id_and_view_direction(make_node):
    node = make_node(caption_topics=['/caption/front', '/caption/left', '/caption/rear'],
                     camera_ids=['front', 'left', 'rear'],
                     camera_yaw_offsets=[0.0, math.pi / 2, math.pi])

    callbacks = {topic: callback for _, topic, callback in node.subscriptions}
    assert set(callbacks) == {'/amcl_pose', '/caption/front', '/caption/left', '/caption/rear'}

    # Robot facing +90deg; each camera's caption should land at the robot
    # yaw plus its mounting offset, wrapped to [-pi, pi].
    node.pose_callback(make_pose(1.0, 2.0, T0, yaw=math.pi / 2))
    callbacks['/caption/front'](String('a desk ahead'))
    callbacks['/caption/left'](String('a plant to the left'))
    callbacks['/caption/rear'](String('a hallway behind'))
    node.memory.milv_wrapper.collection.flush()

    records = node.memory.get_all()
    by_camera = {record.item.camera_id: record.item for record in records}
    assert set(by_camera) == {'front', 'left', 'rear'}

    assert by_camera['front'].caption == 'a desk ahead'
    assert by_camera['front'].theta == pytest.approx(math.pi / 2)
    assert by_camera['left'].theta == pytest.approx(math.pi)
    # pi/2 + pi wraps to -pi/2 rather than growing past pi.
    assert by_camera['rear'].theta == pytest.approx(-math.pi / 2)
    # All cameras share the same robot pose.
    for item in by_camera.values():
        assert item.position == pytest.approx([1.0, 2.0, 0.0])


def test_mismatched_camera_params_are_rejected(node_module, monkeypatch, milvus_db_address):
    monkeypatch.setattr(FakeNode, 'param_overrides', {
        'db_collection': COLLECTION, 'db_ip': milvus_db_address,
        'caption_topics': ['/caption/front', '/caption/rear'],
        'camera_ids': ['front'],
        'camera_yaw_offsets': [0.0, 3.14]})
    with pytest.raises(ValueError, match='same length'):
        node_module.MemoryBuilderNode()


def test_node_pruning_params_reach_the_policy(make_node):
    from remembr.memory.memory_policy import StaleDuplicatePolicy

    node = make_node(enable_memory_pruning=True,
                     pruning_interval=5,
                     pruning_max_age=600.0,
                     pruning_position_radius=2.0,
                     pruning_similarity_threshold=0.8)

    policy = node.memory.policy
    assert isinstance(policy, StaleDuplicatePolicy)
    assert policy.max_age == 600.0
    assert policy.position_radius == 2.0
    assert policy.embedding_similarity_threshold == 0.8
    assert node.memory.prune_every == 5


def test_node_auto_prunes_through_caption_callback(make_node):
    node = make_node(enable_memory_pruning=True,
                     pruning_interval=5,
                     pruning_max_age=600.0,
                     pruning_position_radius=1.0,
                     pruning_similarity_threshold=0.9)

    caption_callback = {topic: callback for _, topic, callback in node.subscriptions}['/caption']

    # Four stale duplicates at the same spot...
    for i in range(4):
        node.pose_callback(make_pose(0.0, 0.0, T0 + i))
        caption_callback(String('i see a desk'))
    # ...then a much newer one; the 5th insert triggers the pruning pass.
    node.pose_callback(make_pose(0.0, 0.0, T0 + 10_000))
    caption_callback(String('i see a desk'))
    node.memory.milv_wrapper.collection.flush()

    records = node.memory.get_all()
    assert len(records) == 1
    assert records[0].item.time == pytest.approx(T0 + 10_000, abs=1.0)
