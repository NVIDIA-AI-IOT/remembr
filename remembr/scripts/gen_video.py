from googleapiclient.errors import HttpError
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import time
import json
import random
import tqdm
import uuid
import numpy as np
import cv2
from time import strftime, localtime
import textwrap
import os
import glob
import pickle as pkl
import argparse
import subprocess


def run_viz(images, positions, map_img, times, delay=3, render=False, captions=None):

    out_images = []

    for i, img in enumerate(images):
        # add a small black bit to the top
        img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, value=0)


        ### Construct time string
        t = localtime(times[i])
        t = strftime('%H:%M:%S', t)
        mins_elapsed = np.round((times[i] - times[0]) / 60, 2)
        time_string = f"Current time: {t}.    Time elapsed: {mins_elapsed} mins"
        cv2.putText(img, time_string, (20,30), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 2, cv2.LINE_AA)# adding timer text


        ### Build the map over time
        # For every pose, let's add a dot to our map. Just be sure to add min x/y to everything
        position = positions[i]
        x = abs(position[0])
        y = abs(position[1])

        map_img = cv2.circle(map_img, (int(x),int(y)), radius=0, color=(255, 0, 0), thickness=-1)

        # x_offset=y_offset=
        # x_offset = img.shape[0] - map_img.shape[0]
        y_offset=50
        x_offset=50
        # small = cv2.resize(map_img, (0,0), fx=0.5, fy=0.5) 
    

        # img[y_offset:y_offset+map_img.shape[0], x_offset:x_offset+map_img.shape[1], 0] = map_img
        # img[y_offset:y_offset+map_img.shape[0], x_offset:x_offset+map_img.shape[1], 1] = map_img
        # img[y_offset:y_offset+map_img.shape[0], x_offset:x_offset+map_img.shape[1], 2] = map_img

        out_images.append(img)
        if render:
            cv2.imshow('viz',img)
            cv2.waitKey(delay)

    return out_images

        

def render_video(args, data, video_start_idx, video_end_idx):

    file_info_dict = {}
    file_info_dict['qa_start_filename'] = data[video_start_idx]['file_start']
    file_info_dict['qa_end_filename'] = data[video_end_idx]['file_end']

    # load the pkl files of segments
    pkl_files = glob.glob(os.path.join(args.coda_dir, str(args.sequence_id), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    positions = []
    images = []
    times = []

    for i in range(0, len(pkl_files)):

        if i % 2 == 0:
            continue

        # let's make a qa_start_idx
        with open(pkl_files[i], 'rb') as f:
            pkl_file = pkl.load(f)

        # captions.append(caption)
        positions.append(pkl_file['position'])
        images.append(pkl_file['cam0'])
        times.append(pkl_file['timestamp'])
    positions = np.array(positions)


    min_x = positions[:,0].min() - 20
    max_x = positions[:,0].max() + 20
    min_y = positions[:,1].min() - 20
    max_y = positions[:,1].max() + 20

    positions[:, 0] = np.abs(positions[:, 0] - min_x)
    positions[:, 1] = np.abs(positions[:, 1] - min_y)

    # normalize

    # map_img = np.ones((int(abs(min_x) + abs(max_x)), int(abs(min_y) + abs(max_y)))) * 255
    map_img = np.ones((int(abs(min_y - max_y)), int(abs(min_x - max_x)))) * 0

    history_viz = run_viz(images, positions, map_img, times, render=False, captions=None)

    return history_viz
    

def save_video(frames, save_path):

    # save video
    codec ='MJPG'
    # fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img = frames[0]
    # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    file = save_path+'.avi'
    video = cv2.VideoWriter(file, fourcc, 15, (img.shape[1], img.shape[0]))


    ### Now let's render them side by side, with the context playing in a loop
    for i in range(len(frames)):
        img = frames[i]
        # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        video.write(img)


        # cv2.imshow('out', img)
        # cv2.waitKey(1)

    video.release()

    mp4_file = file[:-4] + '.mp4'
    # now convert it into a smaller file
    command = f"ffmpeg -i '{os.path.abspath(file)}' -ac 2  -c:a aac -c:v libx264  -vprofile high -bf 0 -strict experimental -f mp4 '{os.path.abspath(mp4_file)}'"
    # proc = os.popen(command)
    # proc.wait()
    subprocess.run(command, shell=True)

    return mp4_file




def main(args):

    seq_id = args.sequence_id

    data_path = os.path.join(args.data_dir, f'captions/{seq_id}/captions/captions_VILA1.5-13b_3_secs.json')
    data = json.load(open(data_path, 'r'))



    all_questions = []

    # In the SQUAD dataset format
    qa_dict = {}


    file_info_dict = {}
    file_info_dict['qa_start_filename'] = data[0]['file_start']
    file_info_dict['qa_end_filename'] = data[-1]['file_end']

    # video_start_time = data[video_start_idx]['time']
    # video_end_time = data[video_end_idx]['time']


    qa_dict['file_info'] = file_info_dict

    # render and save video
    frames = render_video(args, data, 0, len(data)-1)
    os.makedirs(f'./data/videos/{args.sequence_id}/', exist_ok=True)
    local_file_name = save_video(frames, os.path.join(f'./test_slow'))







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Long Horizon Robot QA',
                        description='Runs various LLMs on the QA dataset',)
    
    # data-specific args
    parser.add_argument("--sequence_id", type=int, default=0)
    parser.add_argument('-m', "--model", action="append")


    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument('--coda_dir', type=str, default='./coda_data')


    args = parser.parse_args()
    main(args)