import sys, os
sys.path.append(sys.path[0] + '/..')

import argparse
import gradio as gr

from pymilvus import MilvusClient, DataType

from remembr.memory.milvus_memory import MilvusMemory
from remembr.agents.remembr_agent import ReMEmbRAgent

import subprocess
import threading
# import multiprocessing
import multiprocess as mp 

import torch
torch.multiprocessing.set_start_method('spawn')


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class GradioDemo:


    def __init__(self, args=None):

        if args.rosbag_enabled:
            import rclpy
            rclpy.init()

        self.rosbag_enabled = args.rosbag_enabled

        self.agent = ReMEmbRAgent(llm_type=args.llm_backend)

        self.db_dict = {}

        self.launch_demo()


    def get_options(self, db_uri):
        client = MilvusClient(
            uri=db_uri
        )
        collections = client.list_collections() 

        return gr.Dropdown(choices=collections, label="Select Option", interactive=True)
    
    def update_remembr(self, db_uri, selection):

        ip = db_uri.split('://')[1].split(':')[0]
        memory = MilvusMemory(selection, db_ip=ip)
        self.agent.set_memory(memory)

    def process_file(self, fileobj, upload_name, pos_topic, image_topic, db_uri):
        from chat_demo.db_processor import create_and_launch_memory_builder

        self.db_dict['collection_name'] = upload_name
        self.db_dict['pos_topic'] = pos_topic
        self.db_dict['image_topic'] = image_topic
        self.db_dict['db_ip'] = db_uri.split('://')[1].split(':')[0]


        print("launching threading")
        mem_builder = lambda: create_and_launch_memory_builder(None, db_ip=self.db_dict['db_ip'], \
                                    collection_name=self.db_dict['collection_name'], \
                                        pos_topic=self.db_dict['pos_topic'], \
                                        image_topic=self.db_dict['image_topic'])


        # launch processing thread
        proc = StoppableThread(target=mem_builder)
        proc.start()

        bag_process = subprocess.Popen(["ros2", "bag", "play", fileobj],  stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

        while True:
            ret_code = bag_process.poll()

            if ret_code is not None: 
                print(ret_code, "DONE")
                proc.stop()
                break


    def launch_demo(self):


        # define chatter in here
        def chatter(user_message, history, log):

            # Append user message and response to chat history
            temp_history = history + [(user_message, "...")]
            yield "", temp_history, []
            messages = [(("user", user_message))]
            inputs = {
                "messages": messages
            }
            graph = self.agent.graph
            log = []
            for output in graph.stream(inputs):
                for key, value in output.items():
                    # pprint.pprint(f"Output from node '{key}':")
                    # pprint.pprint("---")
                    # pprint.pprint(value, indent=2, width=80, depth=None)
                    # log += [str(value['messages'])]

                    log.append('------------')
                    log.append(f"Output from node '{key}':")
                    for item in value['messages']:
                        if type(item) == tuple:
                            log.append(item[1])
                        else:
                            if type(item) == str:
                                log.append(item)
                            else: 
                                log.append(item.content)
                            # if len(item.additional_kwargs) > 0:
                                # log.append(str(item.additional_kwargs))

                        log.append('\n')

                    yield "", temp_history, "\n".join(log)

                # pprint.pprint("\n---\n")

            response = output['generate']['messages'][-1]

            out_dict = eval(response)

            # Now let's output only the text output
            response = out_dict['text']

            chat_history = history + [(user_message, response)]

            yield "", chat_history, "\n\n".join(log)


        with gr.Blocks() as demo:

            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot()
                    msg = gr.Textbox(label="Message Input")
                    clear = gr.Button("Clear")
                with gr.Column(scale=1):
                    output_log = gr.Textbox(label="Inference log")


                with gr.Row():
                    # only have this section if we have ROS enabled
                    if self.rosbag_enabled:
                        with gr.Column(scale=1):
                            upload_name = gr.Textbox(label="Name of new DB collection")
                            pos_topic = gr.Textbox(label="Position Topic", value="/amcl_pose")
                            image_topic = gr.Textbox(label="Image Topic", value="front_stereo_camera/left/image_raw")

                            file_upload = gr.File()
                            file_upload.upload(self.process_file, inputs=[file_upload, upload_name, pos_topic, image_topic, db_uri_box])    

                with gr.Column(scale=1):

                    db_uri_box = gr.Textbox(label="Database URI", value="http://127.0.0.1:19530")

                    selector = self.get_options(args.db_uri)
                    with gr.Column(scale=1):
                        refresh = gr.Button("Refresh Options")
                        set = gr.Button("Set Collection")

            msg.submit(chatter, [msg, chatbot], [msg, chatbot, output_log])
            
            refresh.click(lambda x: self.get_options(x), inputs=[db_uri_box], outputs=[selector])
            set.click(self.update_remembr, inputs=[db_uri_box, selector])

        demo.queue(max_size=10)

        demo.launch(server_name=args.chatbot_host_ip, server_port=args.chatbot_host_port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_uri", type=str, default="http://127.0.0.1:19530")
    parser.add_argument("--chatbot_host_ip", type=str, default="localhost")
    parser.add_argument("--chatbot_host_port", type=int, default=7860)

    # Options: 'nim/meta/llama-3.1-405b-instruct', 'gpt-4o', or any Ollama LLMs
    parser.add_argument("--llm_backend", type=str, default='codestral')

    parser.add_argument("--rosbag_enabled", action='store_true')

    args = parser.parse_args()

    demo = GradioDemo(args)