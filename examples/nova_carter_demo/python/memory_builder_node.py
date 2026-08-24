import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from remembr.memory.memory_policy import StaleDuplicatePolicy

from common_utils import format_pose_msg


def wrap_angle(theta: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(theta), math.cos(theta))


class MemoryBuilderNode(Node):

    def __init__(self):
        super().__init__("MemoryBuilderNode")

        self.declare_parameter("db_collection", "test_collection")
        self.declare_parameter("db_ip", "127.0.0.1")

        self.declare_parameter("pose_topic", "/amcl_pose")

        # Multi-camera setup: run one captioner node per camera, each
        # publishing on its own caption topic, and list them here together
        # with a camera id and the camera's mounting yaw relative to the
        # robot base (radians, CCW). Stored memories then carry the camera
        # id, and theta becomes the caption's actual viewing direction, so
        # semantic caption coverage aligns with the sensor FOV. The defaults
        # reproduce the original single front-camera behavior.
        self.declare_parameter("caption_topics", ["/caption"])
        self.declare_parameter("camera_ids", ["front"])
        self.declare_parameter("camera_yaw_offsets", [0.0])

        # Lifelong memory management: periodically drop stale entries whose
        # caption duplicates a nearby, newer observation.
        self.declare_parameter("enable_memory_pruning", False)
        self.declare_parameter("pruning_interval", 100)  # inserts between pruning passes
        self.declare_parameter("pruning_max_age", 3600.0)  # seconds
        self.declare_parameter("pruning_position_radius", 1.0)  # meters
        self.declare_parameter("pruning_similarity_threshold", 0.9)  # caption embedding cosine similarity

        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            self.pose_callback,
            10
        )

        caption_topics = list(self.get_parameter("caption_topics").value)
        camera_ids = list(self.get_parameter("camera_ids").value)
        camera_yaw_offsets = [float(v) for v in self.get_parameter("camera_yaw_offsets").value]
        if not (len(caption_topics) == len(camera_ids) == len(camera_yaw_offsets)):
            raise ValueError(
                "caption_topics, camera_ids, and camera_yaw_offsets must have the same length "
                f"(got {len(caption_topics)}, {len(camera_ids)}, {len(camera_yaw_offsets)})")

        self.caption_subscribers = [
            self.create_subscription(
                String,
                topic,
                self.make_caption_callback(camera_id, yaw_offset),
                10
            )
            for topic, camera_id, yaw_offset
            in zip(caption_topics, camera_ids, camera_yaw_offsets)
        ]
        policy = None
        if self.get_parameter("enable_memory_pruning").value:
            policy = StaleDuplicatePolicy(
                max_age=self.get_parameter("pruning_max_age").value,
                position_radius=self.get_parameter("pruning_position_radius").value,
                embedding_similarity_threshold=self.get_parameter("pruning_similarity_threshold").value,
            )

        self.memory = MilvusMemory(
            self.get_parameter("db_collection").value,
            self.get_parameter("db_ip").value,
            policy=policy,
            prune_every=self.get_parameter("pruning_interval").value
        )

        self.pose_msg = None
        self.caption_msg = None
        self.logger = self.get_logger()

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.pose_msg = msg

    def make_caption_callback(self, camera_id: str, camera_yaw_offset: float):
        def caption_callback(msg: String):
            self.handle_caption(msg, camera_id, camera_yaw_offset)
        return caption_callback

    def handle_caption(self, msg: String, camera_id: str, camera_yaw_offset: float):

        if self.pose_msg is not None:

            position, angle, pose_time = format_pose_msg(self.pose_msg)

            # Store the caption's viewing direction, not the base heading,
            # so retrieved memories point at what the camera actually saw.
            view_theta = wrap_angle(angle + camera_yaw_offset)

            memory = MemoryItem(
                caption=msg.data,
                time=pose_time,
                position=position,
                theta=view_theta,
                camera_id=camera_id
            )

            self.logger.info(f"Added memory item {memory}")

            self.memory.insert(memory)

def main(args=None):
    rclpy.init(args=args)
    node = MemoryBuilderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()