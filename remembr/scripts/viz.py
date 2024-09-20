import json
import numpy as np

from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import PromptTemplate
from time import strftime, localtime
import numpy as np
import tqdm

import re
import time
import uuid

import argparse
import glob
import pickle as pkl

import os, sys
import cv2
import textwrap

# load this directory
sys.path.append(sys.path[0] + '/..')

def run_viz(images, positions, map_img, times, question, delay=3, render=False, captions=None):

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

        ### Construct question string
        wrap_i = 0
        if captions is None:
            wrapped_text = textwrap.wrap(question, width=60)
        else:
            wrapped_text = textwrap.wrap(captions[i], width=60)

        num_lines = 10
        img = cv2.copyMakeBorder(img, 0, 40*num_lines, 0, 0, cv2.BORDER_CONSTANT, None, value=0)


        for line in wrapped_text:
            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

            gap = textsize[1] + 5

            y = int((img.shape[0])) + wrap_i * gap - 40*(num_lines-1)
            x = 20 

            cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, 
                        (255,255,255), 
                        2, 
                        lineType = cv2.LINE_AA)
            wrap_i +=1


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
    

        img[y_offset:y_offset+map_img.shape[0], x_offset:x_offset+map_img.shape[1], 0] = map_img
        img[y_offset:y_offset+map_img.shape[0], x_offset:x_offset+map_img.shape[1], 1] = map_img
        img[y_offset:y_offset+map_img.shape[0], x_offset:x_offset+map_img.shape[1], 2] = map_img

        out_images.append(img)
        if render:
            cv2.imshow('viz',img)
            cv2.waitKey(delay)

    return out_images

        



    


def visualize(args, data, question_idx, render=False):


    # first get the file paths
    # get a list of videos for the question and for the ideal context
    qa_instance = data[question_idx]

    question = qa_instance['question']
    context = qa_instance['context']
    start_time = qa_instance['start_time']
    answers = qa_instance['answers']
    id = qa_instance['id']

    file_info_dict = qa_instance['file_info']

    pkl_files = glob.glob(os.path.join(args.coda_dir, str(args.sequence_id), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    file_times = np.array([float(file.split('/')[-1][:-4]) for file in pkl_files])

    ### Load captions
    caption_path = os.path.join(args.data_dir, str(args.sequence_id), 'captions.json')
    caption_list = json.load(open(caption_path, 'r'))

    caption_start_ids = [item['id'] for item in caption_list]
    caption_start_ids.sort(key=lambda x: float(x.split('/')[-1][:-4])) # should already be sorted
    caption_times = np.array([float(file.split('/')[-1][:-4]) for file in caption_start_ids])

    ### Visualize the whole question first
    qa_start_path = os.path.join(args.coda_dir, str(args.sequence_id), file_info_dict['qa_start_filename'])
    qa_end_path = os.path.join(args.coda_dir, str(args.sequence_id), file_info_dict['qa_end_filename'])

    qa_start_idx = pkl_files.index(qa_start_path)
    qa_end_idx = pkl_files.index(qa_end_path)

    positions = []
    images = []
    times = []
    captions = []


    for i in range(qa_start_idx, qa_end_idx):

        if i % 2 == 0:
            continue

        # let's make a qa_start_idx
        with open(pkl_files[i], 'rb') as f:
            pkl_file = pkl.load(f)

        current_time = file_times[i]
        # find the first negative number
        diff = caption_times - current_time
        caption_idx = np.argmax(diff > 0) # maybe subtract 1
        caption = caption_list[caption_idx]['caption']

        captions.append(caption)
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


    history_viz = run_viz(images, positions, map_img, times, question, render=render, captions=captions)


    ### Visualize the context required to answer the question 

    context_start_path = os.path.join(args.coda_dir, str(args.sequence_id), file_info_dict['context_start_filename'])
    context_end_path = os.path.join(args.coda_dir, str(args.sequence_id), file_info_dict['context_end_filename'])

    q_start_idx = pkl_files.index(context_start_path)
    q_end_idx = pkl_files.index(context_end_path)

    positions = []
    images = []
    times = []


    for i in range(q_start_idx, q_end_idx):
        # let's make a qa_start_idx
        with open(pkl_files[i], 'rb') as f:
            pkl_file = pkl.load(f)


        positions.append(pkl_file['position'])
        images.append(pkl_file['cam0'])
        times.append(pkl_file['timestamp'])
    positions = np.array(positions)
    map_img = np.ones((int(abs(min_y - max_y)), int(abs(min_x - max_x)))) * 0

    context_viz = run_viz(images, positions, map_img, times, question, delay=30, render=render)


    ### Visualize the answer, if a model was specified
    model_outs = []
    if len(args.model) > 0:
        for model in args.model:

            out_path = os.path.join(args.out_dir, str(args.sequence_id), f'{args.qa_filename}', f'{model}.json')

            if model and os.path.exists(out_path):

                pred_data = json.load(open(out_path, 'r'))
                pred_data = pred_data['responses']

                response = pred_data[question_idx]["reasoning"]
                answer = pred_data[question_idx]["error"]
                # is_correct = pred_data[question_idx]["correct"]

                response = f"Model: {model} \nReasoning: {response} \nError: {answer}"
            else:
                response = ""
            
            model_outs.append(response)

    response = "\n\n".join(model_outs)
                
    # Write an image the length of images
    text_image = np.zeros_like(context_viz[0])

    wrap_i = 0
    wrapped_text = '\n'.join(textwrap.wrap(response, width=60, replace_whitespace=False)).split('\n')

    for line in wrapped_text:
        textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

        gap = textsize[1] + 5

        # y = int((text_image.shape[0])) + wrap_i * gap - 40*(len(wrapped_text) -1)
        # x = 20 

        y = int(40) + wrap_i * gap
        # x = int((text_image.shape[1] - textsize[0]) / 2)
        x = 20

        cv2.putText(text_image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (255,255,255), 
                    2, 
                    lineType = cv2.LINE_AA)
        wrap_i +=1


    # slow down the output of context_viz by repeating each element
    K = 5
    context_viz = [ele for ele in context_viz for i in range(K)]


    img = cv2.hconcat([history_viz[0], context_viz[0], text_image])
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    size = img.shape

    codec ='MJPG'
    # fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video = cv2.VideoWriter('video.avi', fourcc, 60, (img.shape[1], img.shape[0]))


    ### Now let's render them side by side, with the context playing in a loop
    for i in range(len(history_viz)):
        history_img = history_viz[i % len(history_viz)]
        context_img = context_viz[i % len(context_viz)]

        img = cv2.hconcat([history_img, context_img, text_image])

        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        video.write(img)


        cv2.imshow('out', img)
        cv2.waitKey(1)

    video.release()

    



def main(args):

    data_path = os.path.join(args.data_dir, str(args.sequence_id), 'questions', f'{args.qa_filename}.json')

    data = json.load(open(data_path, 'r'))
    data = data['data']

    visualize(args, data, args.question_idx)

    import pdb; pdb.set_trace()
    exit()

    running_successes = 0
    responses = []
    for i in tqdm.tqdm(range(0, len(data)), total=len(data)):

        qa_instance = data[i]
        question = qa_instance['question']
        context = qa_instance['context']
        start_time = qa_instance['start_time']
        answers = qa_instance['answers']
        id = qa_instance['id']

        answer = answers['text'][1] # for now, this index is guaranteed to say Yes/No


        if 'remembr' in args.model:
            context = "" # we don't use any additional context since we will retrieve it

        out_dict = answer_squad_question(args, chain, question, answer, context=context)


        out_dict['question'] = qa_instance['question']
        out_dict['id'] = id

        if out_dict['correct']:
            running_successes += 1

        print("Question:", question)
        print("Response:", out_dict['response'])
        print("Correct?", out_dict['correct'])
        print("Running Accuracy", running_successes/(i+1))
        print()

        responses.append(running_successes)


    # # save all_questions into json
    out_json = {
        "version": 0.1,
        "responses": responses
    }

    # save the outputs
    # out_path = os.path.join(args.data_dir, args.sequence_id, f'{args.qa_filename}.json')
    # os.makedirs(out_path, exist_ok=True)
    # with open(os.join(out_path, f'{args.model}.json'), 'w') as f:
    #     # to_save = json.dumps(out_json, indent=4)
    #     json.dump(out_json, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Long Horizon Robot QA',
                        description='Runs various LLMs on the QA dataset',)
    
    # data-specific args
    parser.add_argument("--sequence_id", type=int, default=0)
    parser.add_argument("--question_idx", type=int, default=0)
    parser.add_argument("--qa_filename", type=str, default='qa')

    parser.add_argument('-m', "--model", action="append")


    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--out_dir", type=str, default="./out/")
    parser.add_argument('--coda_dir', type=str, default='./coda_data')

    # all model args
    # parser.add_argument("--use_gt_context", type=bool, default=False)


    # llm-specific args
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num_ctx", type=int, default=8192)

    # remembr specific args
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--db_name", type=str, default='test')


    args = parser.parse_args()
    main(args)