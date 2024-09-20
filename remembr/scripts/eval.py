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
import sys
import os, sys
import pickle as pkl
from PIL import Image as PILImage
import glob

from dataclasses import asdict
import argparse
import traceback 

# load this directory
sys.path.append(sys.path[0] + '/..')

from agents.remembr_agent import ReMEmbRAgent
from agents.non_agent import NonAgent
from agents.vlm_non_agent import VLMNonAgent

from memory.memory import MemoryItem
from memory.milvus_memory import MilvusMemory
from memory.text_memory import TextMemory
from memory.video_memory import VideoMemory, ImageMemoryItem

from tools.tools import format_docs


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)

# we can have binary, position-based, time-based, or description-based. let's answer accordingly
def evaluate_output(qa_instance, predicted):

    out_error = {}

    q_type = qa_instance['type']
    if 'position' in q_type:

        answer = np.array(qa_instance['answers']['position'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['position']) == str:
            predicted['position'] = eval(predicted['position'])
        pred_pos = np.array(predicted['position'])

        dist = np.linalg.norm(answer - pred_pos)

        out_error['position_error'] = dist

    elif 'binary' in q_type:

        answer = qa_instance['answers']['text'][1] # we made this assumption in other examples that binary answer is the second one

        if 'binary' in predicted and (predicted['binary'].lower() == "yes" or predicted['binary'].lower() == "no"):
            # get correct/incorrect label
            if predicted['binary'].lower() == answer.lower():
                correct = 1
            else:
                correct = 0

            out_error['binary_iscorrect'] = correct

    elif 'time' in q_type:

        answer = np.array(qa_instance['answers']['time'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['time']) == str:
            predicted['time'] = eval(predicted['time'])
        pred_time = np.array(predicted['time'])

        dist = abs(answer - pred_time)

        out_error['time_error'] = dist

    elif 'duration' in q_type:

        answer = np.array(qa_instance['answers']['duration'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['duration']) == str:
            predicted['duration'] = eval(predicted['duration'])
        pred_time = np.array(predicted['duration'])

        dist = abs(answer - pred_time)

        out_error['duration_error'] = dist

    elif 'text' in q_type:
        answer = qa_instance['answers']['text']
        out_error = {'answer': answer}

    else:
        raise Exception("We do not support question type " + q_type)

    return out_error


def answer_squad_question(model, question, qa_instance):

    print(f'Question: {question}')

    parsed = None
    while True:
        try:

            start_time = time.time()
            response = model.query(question)
            end_time = time.time()

            elapsed = end_time - start_time

            parsed = asdict(response)

            out_error = evaluate_output(qa_instance, parsed)
            print("Time elapsed", elapsed)

        except Exception as e:
            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info()) 
            continue

        return_dict = {"response": parsed}
        return_dict.update(parsed)
        return_dict['error'] = out_error
        return_dict['elapsed'] = elapsed

        return return_dict


def load_memory(args, qa_instance, use_milvus=True, use_optimal_context=False, ip_address='127.0.0.1'):
    # Here we load everything needed to load a MilvusDB instance neatly
    start_time = qa_instance['start_time']
    end_time = qa_instance['end_time']


    if use_milvus:
        # milv = MilvusWrapper(ip_address=ip_address)
        memory = MilvusMemory(f"eval_memory_{args.sequence_id}", db_ip=ip_address, time_offset=start_time)
    elif 'vlm' in args.model:
        memory = VideoMemory()
    else:
        memory = TextMemory()

    memory.reset()


    captions_path = os.path.join(args.data_dir, 'captions', str(args.sequence_id), 'captions', f'{args.caption_file}.json')

    with open(captions_path, 'r') as f:
        out = json.load(f)


    outputs = []

    # Compute start idx
    all_start_times = np.array([float(x['file_start'][:-4]) for x in out])
    diff = all_start_times - start_time
    start_idx = np.argmin(np.abs(diff))

    # Compute end idx
    all_end_times = np.array([float(x['file_end'][:-4]) for x in out])
    diff = all_end_times - end_time
    end_idx = np.argmin(np.abs(diff))


    pkl_files = glob.glob(os.path.join(args.coda_dir, str(args.sequence_id), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))

    for i in range(start_idx, end_idx+1):

        item = out[i]
        entity = {
            'position': item['position'],
            'theta': item['theta'], # ignoring rotation
            'time': item['time'], 
            'caption': item['caption'],
        }

        outputs.append(entity)

        if type(memory) == VideoMemory:

            qa_start_path = os.path.join(args.coda_dir, str(args.sequence_id), out[i]['file_start'])
            qa_end_path = os.path.join(args.coda_dir, str(args.sequence_id), out[i+1]['file_start'])

            qa_start_idx = pkl_files.index(qa_start_path)
            qa_end_idx = pkl_files.index(qa_end_path)
            idxs = np.linspace(qa_start_idx, qa_end_idx, 6, dtype=int)

            for pkl_idx in idxs:
                # pkl_path = os.path.join(args.coda_dir, str(args.sequence_id), item['file_start'])
                pkl_path = pkl_files[pkl_idx]
                with open(pkl_path, 'rb') as f:
                    pkl_data = pkl.load(f)
                entity['image'] = PILImage.fromarray(pkl_data['cam0'].astype('uint8'), 'RGB')

            entity = ImageMemoryItem.from_dict(entity)
        else:
            entity = MemoryItem.from_dict(entity)

        if use_milvus:
            memory.insert(entity, text_embedding=item['text_embedding'])
        else:
            memory.insert(entity)

    if use_optimal_context:
        # then replace the full memory with the optimal context
        memory = TextMemory()
        memory.insert(qa_instance['context'])


    return memory, outputs

def main(args):

    use_milvus = False
    use_optimal_context = False
    if 'remembr' in args.model:
        base_llm = args.model.split('+')[-1]
        agent = ReMEmbRAgent(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_milvus = True

    elif 'optimal' in args.model:
        base_llm = args.model.split('+')[-1]
        agent = NonAgent(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_optimal_context = True
    elif 'vlm' in args.model:
        agent = VLMNonAgent(llm_type='gpt-4o')

    else:
        agent = NonAgent(llm_type=args.model, num_ctx=args.num_ctx*4, temperature=args.temperature)


    data_path = os.path.join(args.data_dir, 'questions', str(args.sequence_id), args.qa_file+'.json')

    data = json.load(open(data_path, 'r'))
    data = data['data']


    running_successes = 0
    num_binary = 0

    running_pos_error = 0
    num_position = 0

    running_time_error = 0
    num_time = 0

    running_duration_error = 0
    num_duration = 0
    
    responses = []
    for i in tqdm.tqdm(range(0, len(data)), total=len(data)):

        print(f"Evaluating {i} out of {len(data)}")

        qa_instance = data[i]
        question = qa_instance['question']
        context = qa_instance['context']
        start_time = qa_instance['start_time']
        answers = qa_instance['answers']
        id = qa_instance['id']

        if (qa_instance['type'] == 'text'):
            print("Skipping text questions for now")
            responses.append({}) # this means skipped!
            continue

        memory, instance_captions = load_memory(args, data[i], use_milvus=use_milvus, use_optimal_context=use_optimal_context, ip_address=args.db_ip)
        if len(instance_captions) == 0: # ISSUE
            print("Length of Instance Captions is 0. It should not be")
            import pdb; pdb.set_trace()

        print("HISTORY LENGTH", len(instance_captions))

        # model.update_for_instance(captions=instance_captions, ref_time=start_time)
        agent.set_memory(memory)


        out_dict = answer_squad_question(agent, question, qa_instance)


        out_dict['question'] = qa_instance['question']
        out_dict['id'] = id


        error_dict = out_dict['error']

        # keep track of how many of each. usually all CSVs are one type only
        if qa_instance['type'] == 'position':
            num_position += 1
            if 'position_error' in error_dict:
                running_pos_error += error_dict['position_error']
        elif qa_instance['type'] == 'binary':
            num_binary += 1
            if 'binary_iscorrect' in error_dict:
                running_successes += error_dict['binary_iscorrect']
        elif qa_instance['type'] == 'time':
            num_time += 1
            if 'time_error' in error_dict:
                running_time_error += error_dict['time_error']
        elif qa_instance['type'] == 'duration':
            num_duration += 1
            if 'duration_error' in error_dict:
                running_duration_error += error_dict['duration_error']

        print("Question:", question)
        if 'response' in out_dict:
            print("Response:", out_dict['response'])
        print("Running Binary QA accuracy", running_successes/(num_binary+1))
        print("Running Spatial Error", running_pos_error/(num_position+1))
        print("Running Temporal Error", running_time_error/(num_time+1))
        print("Running Duration Error", running_duration_error/(num_duration+1))

        print()


        responses.append(out_dict)


    # save all_questions into json
    out_json = {
        "version": 0.1,
        "responses": responses
    }

    # save the outputs
    out_path = os.path.join(args.out_dir, str(args.sequence_id), args.qa_file)
    os.makedirs(out_path, exist_ok=True)

    name = args.model+'__'+args.caption_file+args.postfix
    with open(os.path.join(out_path, f'{name}.json'), 'w') as f:
        # to_save = json.dumps(out_json, indent=4)
        json.dump(out_json, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Long Horizon Robot QA',
                        description='Runs various LLMs on the QA dataset',)
    
    # data-specific args
    parser.add_argument("--sequence_id", type=int, default=0)
    parser.add_argument("--model", type=str, default="remembr+llama3")
    parser.add_argument("--qa_file", type=str, default="human_qa")
    parser.add_argument("--caption_file", type=str, default="captions_VILA1.5-13b_3_secs")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--coda_dir", type=str, default="./coda_data/")

    parser.add_argument("--out_dir", type=str, default="./out/")

    parser.add_argument("--postfix", type=str, default='_0')


    # all model args
    # parser.add_argument("--use_gt_context", type=bool, default=False)


    # llm-specific args
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_ctx", type=int, default=8192*8)

    # remembr specific args
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--db_name", type=str, default='test')
    parser.add_argument("--db_ip", type=str, default='127.0.0.1')


    args = parser.parse_args()
    main(args)