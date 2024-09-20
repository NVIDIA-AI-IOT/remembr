import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.clock import Clock
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

import numpy as np
from scipy.spatial.transform import Rotation as R

import cv2
from cv_bridge import CvBridge
bridge = CvBridge()

import message_filters
import collections
import argparse

import time
from PIL import Image as PILImage

import sys

from remembr.captioners.vila_captioner import VILACaptioner
from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from PIL import Image as im

import concurrent


def memory_builder_args(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-3b")
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, default="<video>\n Please describe what you see in the few seconds of the video.")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    return args

def create_and_launch_memory_builder(args=None, **kwargs):
    mem_args = memory_builder_args()
    print(kwargs)
    mem_builder = ROSMemoryBuilder(mem_args, **kwargs)

    print("Built memory builder. Starting spin")
    mem_builder.spin()

    mem_builder.destroy_node()
    rclpy.shutdown()

class ROSMemoryBuilder(Node):

    def __init__(self, args, segment_time=3, \
                image_topic='/front_stereo_camera/left/image_raw', \
                collection_name="test", db_ip="127.0.0.1", \
                pos_topic='/amcl_pose', queue_size=1000):
        
        super().__init__('minimal_subscriber')

    
        self.start_time = 0
        self.segment_time = segment_time
        self.data_buffer = []

        self.vila_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.data_buffer = [[]]
        self.pose_buffer = []
        self.image_buffer = []
        self.last_pose = [0,0,0]


        self.is_compressed = False
        if 'compressed' in image_topic:
            self.img_sub = self.create_subscription(
                CompressedImage,
                image_topic,
                self.img_listener_callback,
                10)
            self.is_compressed = True
        else:
            self.img_sub = self.create_subscription(
                Image,
                image_topic,
                self.img_listener_callback,
                10)

        if 'odom' in pos_topic:
            self.pose_sub = self.create_subscription(
                Odometry,
                pos_topic,
                self.pose_listener_callback,
                10)
        else:
            self.pose_sub = self.create_subscription(
                PoseWithCovarianceStamped,
                pos_topic,
                self.pose_listener_callback,
                10)

        self.counter = 0

        self.captioner = VILACaptioner(args)
        self.memory = MilvusMemory(collection_name, db_ip=db_ip)

    def spin(self):
        print("STARTING SPIN")
        rclpy.spin(self)


    def pose_listener_callback(self, odom_msg):
        # self.get_logger().info('I heard: "%s"' % msg.data)
        print("Got a synchronized message")
        # we can also only accept every third message
        position = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
        quat = np.array([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z,odom_msg.pose.pose.orientation.w])
        euler_rot_z = R.from_quat(quat).as_euler('xyz')[-1] # take z rotation
        stamp = odom_msg.header.stamp
        converted_time = float(str(stamp.sec) + '.' + str(stamp.nanosec))
        data_dict = {
            'position': position,
            'orientation': euler_rot_z,
            'time': converted_time
        }
        self.last_pose = data_dict
        self.pose_buffer.append(data_dict)

    def img_listener_callback(self, img_msg):
        # self.get_logger().info('I heard: "%s"' % msg.data)
        self.counter += 1
        if self.counter % 10 != 0: # get every 10th image here
            return

        if self.is_compressed:
            cv_img = bridge.compressed_imgmsg_to_cv2(img_msg)
        else:
            cv_img = bridge.imgmsg_to_cv2(img_msg)

        PIL_img = im.fromarray(cv_img)
        stamp = img_msg.header.stamp
        converted_time = float(str(stamp.sec) + '.' + str(stamp.nanosec))
        data_dict = {
            'image': PIL_img,
            'time': converted_time
        }

        self.image_buffer.append(data_dict)

        if converted_time - self.start_time >= self.segment_time:
            before_time = time.time()
            pose_dict = {}

            if len(self.pose_buffer) == 0:
                pose_dict = {
                    'position': self.last_pose['position'],
                    'orientation': self.last_pose['orientation'],
                }
            else:

                positions = []
                orientations = []
                for item in self.pose_buffer:
                    positions.append(item['position'])
                    orientations.append(item['orientation'])
                positions = np.array(positions)
                positions = np.mean(positions, axis=0)
                orientations = np.array(orientations)
                orientations = np.mean(orientations, axis=0)
                pose_dict = {
                    'position': positions,
                    'orientation': orientations
                }
            self.vila_executor.submit(self.process_into_db(self.image_buffer, pose_dict))
            after_time = time.time()
            print("Time to compute =", after_time - before_time)
            self.data_buffer = [[]]
            self.start_time = converted_time # TODO: Check if this is valid
            self.image_buffer = []
            self.pose_buffer = []

    def process_into_db(self, image_buffer, pose_dict):
        print('******************* Processing', len(image_buffer))
        images = []
        positions = []
        orientations = []
        for item in image_buffer:
            images.append(item['image'])
        # TODO: downsample to 6 images
        positions = pose_dict['position']
        orientations = pose_dict['orientation']

        out_text = out_text = self.captioner.caption(images)
        print(out_text)

        mid_time = (image_buffer[0]['time'] + image_buffer[-1]['time'])/2

        entity = {
            'position': [positions[0], positions[1], positions[2]],
            'theta': orientations,
            'time': mid_time,
            'caption': out_text,
        }

        entity = MemoryItem.from_dict(entity)
        self.memory.insert(entity)




def main(args=None):
    print("Starting")

    args = memory_builder_args(args)

    mem_builder = ROSMemoryBuilder(args)

    rclpy.spin(mem_builder)

    mem_builder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()