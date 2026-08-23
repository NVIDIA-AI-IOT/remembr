import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from scipy.spatial.transform import Rotation as R


def format_pose_msg(msg: PoseWithCovarianceStamped):

    position = np.array([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z
    ])

    quat = np.array([
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    ])

    euler_rot_z = R.from_quat(quat).as_euler('xyz')[-1] # take z rotation

    stamp = msg.header.stamp
    converted_time = float(str(stamp.sec) + '.' + str(stamp.nanosec))

    return position, euler_rot_z, converted_time
