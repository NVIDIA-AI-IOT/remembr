import os
import argparse
import json
import select
from os.path import join

import numpy as np
from scipy.spatial.transform import Rotation as R

# import rospy
from cv_bridge import CvBridge
import cv2
import tf2_ros
import tf.transformations as tf_trans

from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import  Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Point

import pickle as pkl
from tqdm import tqdm 

# For imports
import sys
CODA_DEVKIT_DIRECTORY = os.path.join(os.environ['CODA_ROOT_DIR'], '..')
sys.path.append(CODA_DEVKIT_DIRECTORY)

from helpers.visualization import (clear_marker_array, create_3d_bbox_marker, pub_pose,
                                    project_3dbbox_image, pub_pc_to_rviz, apply_semantic_cmap,
                                    apply_rgb_cmap, project_3dpoint_image)
from helpers.calibration import load_extrinsic_matrix, load_camera_params
from helpers.sensors import (set_filename_dir, read_bin, read_sem_label)
from helpers.geometry import pose_to_homo
from helpers.constants import *
from scripts.check_stereo_rgb import extract_ts

from scripts.gen_pc_for_js import (downsample_point_cloud, save_bin_file, read_bbox_file, save_bbox_file)

# from helpers.ros_visualization import publish_3d_bbox

import sys
import termios
import tty

parser = argparse.ArgumentParser(description="CODa rviz visualizer")
parser.add_argument("-s", "--sequence", type=str, default="0", 
                    help="Sequence number (Default 0)")
parser.add_argument("-f", "--start_frame", type=str, default="0",
                    help="Frame to start at (Default 0)")
parser.add_argument("-c", "--color_type", type=str, default="classId", 
                    help="Color map to use for coloring boxes Options: [isOccluded, classId] (Default classId)")
parser.add_argument("-l", "--log", type=str, default="",
                    help="Logs point cloud and bbox annotations to file for external usage")
parser.add_argument("-n", "--namespace", type=str, default="coda",
                    help="Select a namespace to use for published topics")

def get_key():
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return ch

def vis_annos_rviz(args):
    indir = os.getenv(ENV_CODA_ROOT_DIR)
    assert indir is not None, f'Directory for CODa cannot be found, set {ENV_CODA_ROOT_DIR}'
    sequence, start_frame, color_type, log_dir = args.sequence, int(args.start_frame), args.color_type, \
        args.log
    namespace = args.namespace
    # rospy.init_node('CODa_publisher')

    # for sequence in sequences:
    # Path to the data
    calib_dir       = join(indir, CALIBRATION_DIR, sequence)
    lidar_ts_dir    = join(indir, TIMESTAMPS_DIR,
                                    f"{sequence}.txt")
    poses_dir       = join(indir, POSES_DIR, "dense")
    
    # Pose DATA
    pose_file   = join(poses_dir, f'{sequence}.txt')
    pose_np     = np.fromfile(pose_file, sep=' ').reshape(-1, 8)

    lidar_ts_np = np.loadtxt(lidar_ts_dir, dtype=np.float64)
    
    # Calibration DATA (Extrinsic and Intrinsic)
    os1_to_base_ext_file = join(calib_dir, "calib_os1_to_base.yaml")
    os1_to_cam0_ext_file = join(calib_dir, "calib_os1_to_cam0.yaml")
    os1_to_cam1_ext_file = join(calib_dir, "calib_os1_to_cam1.yaml")

    cam0_intrinsics_file = join(calib_dir, "calib_cam0_intrinsics.yaml")
    cam1_intrinsics_file = join(calib_dir, "calib_cam1_intrinsics.yaml")
    
    os1_to_base_ext = load_extrinsic_matrix(os1_to_base_ext_file) 
    os1_to_cam0_ext = load_extrinsic_matrix(os1_to_cam0_ext_file)
    os1_to_cam1_ext = load_extrinsic_matrix(os1_to_cam1_ext_file)

    cam0_K, cam0_D, cam0_size = load_camera_params(cam0_intrinsics_file)
    cam1_K, cam1_D, cam1_size = load_camera_params(cam1_intrinsics_file)

    # Stereo Depth DATA
    stereo_img_dir = set_filename_dir(indir, TRED_RAW_DIR, "cam3", sequence)
    if os.path.exists(stereo_img_dir):
        stereo_img_files = np.array([img_file for img_file in os.listdir(stereo_img_dir) if img_file.endswith('.png')])
        stereo_img_ts = np.array([extract_ts(img_file) for img_file in stereo_img_files])
        stereo_sort_indices = np.argsort(stereo_img_ts) # low to high
        stereo_img_files = stereo_img_files[stereo_sort_indices]
        stereo_img_ts = stereo_img_ts[stereo_sort_indices]

    pose_skip = 1
    last_frame = 0
    for pose_idx, pose in tqdm(enumerate(pose_np), total=len(pose_np)):
        if pose_idx % pose_skip != 0:
            continue
        # print(f'Publishing sensor info at pose {pose_idx}')
        # Get Pose
        pose_ts, x, y, z, qw, qx, qy, qz = pose
        base_pose = np.eye(4)
        try:
            base_pose[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            base_pose[:3, 3] = [x, y, z]
        except:
            continue # skip if it doesn't work

        # POINTS STORED AS xyz


        # Get the closest frame
        frame = np.searchsorted(lidar_ts_np, pose_ts, side='left')

        if frame == last_frame:
            continue

        # lidar_ts = rospy.Time.from_sec(lidar_ts_np[frame])
        last_frame = frame

        if frame < start_frame:
            continue

        # Broadcast TF (odom -> os1)

        try:
            rotation = tf_trans.quaternion_from_matrix(base_pose)
        except:
            continue # skip if it doesn't work
        
        # Get the path to the data
        pc_file   = set_filename_dir(indir, TRED_COMP_DIR, "os1", sequence, frame, include_name=True)
        cam0_file = set_filename_dir(indir, TWOD_RECT_DIR, "cam0", sequence, frame, include_name=True)
        cam1_file = set_filename_dir(indir, TWOD_RECT_DIR, "cam1", sequence, frame, include_name=True)
        bbox_file = set_filename_dir(indir, TRED_BBOX_LABEL_DIR, "os1", sequence, frame, include_name=True)
        sem_file = set_filename_dir(indir, SEMANTIC_LABEL_DIR, "os1", sequence, frame, include_name=True)


        # Publish the 3D Bounding Box
        if os.path.exists(bbox_file):
            bbox_3d_json = json.load(open(bbox_file, 'r'))

            # Draw 3d Bounding Box 
            bbox_3d_markers = []


            for bbox_3d in bbox_3d_json['3dbbox']:
                
                bbox_3d_color = (0, 0, 0, 1.0) # Black by default
                if color_type=="isOccluded":
                    color_id = OCCLUSION_TO_ID[bbox_3d['labelAttributes']['isOccluded']]
                    bbox_3d_color = OCCLUSION_ID_TO_COLOR[color_id]
                elif color_type=="classId":
                    color_id = BBOX_CLASS_TO_ID[bbox_3d['classId']]
                    bbox_3d_color_scaled_bgr = [c/255.0 for c in BBOX_ID_TO_COLOR[color_id] ] + [1]
                    bbox_3d_color_scaled_rgb = [
                        bbox_3d_color_scaled_bgr[2], bbox_3d_color_scaled_bgr[1], bbox_3d_color_scaled_bgr[0]
                    ]
                    bbox_3d_color = (bbox_3d_color_scaled_rgb)

            # Log 3d Bbox for external viewing
            if log_dir!="":
                bbox_outpath = f'{log_dir}/{last_frame}bbox.bin'
                save_bbox_file(read_bbox_file(bbox_file), bbox_outpath)
                print(f'Logged bbox frame {last_frame} to {bbox_outpath}')
                    

        # Publish Camera Images
        if os.path.exists(cam0_file) and os.path.exists(cam1_file):
            # Camera 0
            cam0_image = cv2.imread(cam0_file, cv2.IMREAD_COLOR)
            # Camera 1
            cam1_image = cv2.imread(cam1_file, cv2.IMREAD_COLOR)

            # Project 3D Bounding Box to 2D
            if os.path.exists(bbox_file):
                bbox_3d_json = json.load(open(bbox_file, 'r'))
            else:
                bbox_3d_json = {}


        # Publish Stereo Depth Image if it exists
        if os.path.exists(stereo_img_dir):
            closest_stereo_img_idx = np.searchsorted(stereo_img_ts, pose_ts, side='left')
            closest_stereo_file = stereo_img_files[closest_stereo_img_idx]
            closest_stereo_path = join(stereo_img_dir, closest_stereo_file)
            stereo_img_np = cv2.imread(closest_stereo_path, cv2.IMREAD_GRAYSCALE)
            max_val = np.max(stereo_img_np)

        # we want to save:
        # cam0, stereo, bbox_3d_json, pose, and lidar_ts

        out_dict = {}
        out_dict['cam0'] = cam0_image
        out_dict['bbox_3d'] = bbox_3d_json
        out_dict['position'] = np.array([x,y,z])
        out_dict['rotation'] = np.array(rotation)
        out_dict['timestamp'] = pose_ts


        # Make trajectory here
        output_dir = join('coda_data', sequence)
        if not os.path.exists(output_dir):
            print("Output image dir for %s does not exist, creating..."%output_dir)
            os.makedirs(output_dir)

        with open(join(output_dir, f'{pose_ts}.pkl'), 'wb') as handle:
            pkl.dump(out_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parser.parse_args()
    vis_annos_rviz(args)
