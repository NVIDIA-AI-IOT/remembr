import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory

from common_utils import format_pose_msg



class MemoryBuilderNode(Node):

    def __init__(self):
        super().__init__("MemoryBuilderNode")

        self.declare_parameter("db_collection", "test_collection")
        self.declare_parameter("db_ip", "127.0.0.1")

        self.declare_parameter("pose_topic", "/amcl_pose")
        self.declare_parameter("caption_topic", "/caption")

        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            self.pose_callback,
            10
        )

        self.caption_subscriber = self.create_subscription(
            String,
            self.get_parameter("caption_topic").value,
            self.query_callback,
            10
        )
        self.memory = MilvusMemory(
            self.get_parameter("db_collection").value,
            self.get_parameter("db_ip").value
        )

        self.pose_msg = None
        self.caption_msg = None
        self.logger = self.get_logger()

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.pose_msg = msg

    def caption_callback(self, msg: String):

        if self.pose_msg is not None:

            position, angle, pose_time = format_pose_msg(self.pose_msg)

            memory = MemoryItem(
                caption=msg.data,
                time=pose_time,
                position=position,
                theta=angle
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