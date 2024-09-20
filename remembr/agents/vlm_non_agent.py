import json
import numpy as np
import sys, os
import re
import base64, io
from PIL import Image
from time import strftime, localtime

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage 

sys.path.append(sys.path[0] + '/..')
from tools.tools import format_docs
from utils.util import file_to_string

from remembr.agents.agent import Agent, AgentOutput
from remembr.memory.memory import Memory
from remembr.memory.video_memory import VideoMemory, ImageMemoryItem


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    parsed = json.loads(parsed)
    return parsed

def np_image_to_base64(image):

    image = Image.fromarray(np.uint8(image))
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def construct_messages_from_memory(memory_items: ImageMemoryItem):

    messages = []
    
    for i, item in enumerate(memory_items):
            
            t = item.time
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            text = f"Frame {i} at time={t}, the robot was at an average position of {np.array(item.position).round(3).tolist()}"
            text += f"with an average orientation of {round(item.theta, 3)} radians."
            
            # if item.caption != "":
            #     text+= f"Description of Frame {i}: {item.caption}"

            text += "Image robot saw:"
            base64_image = np_image_to_base64(item.image)
            text_content = {
                "type": "text", 
                "text": text
            }

            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": 'low',
                },
            }
            messages.append(text_content)
            messages.append(image_content)

    print(len(messages)//2)

    return HumanMessage(content=messages)


class VLMNonAgent(Agent):
    def __init__(self, llm_type='llama3', num_ctx=8192, temperature=0):
        
        self.llm_type = llm_type

        if 'gpt-4' in 'llm_type':
            # TODO: ADD OpenAI here
            pass
        else:
            raise NotImplementedError
        
        top_level_path = str(os.path.dirname(__file__)) + '/../'
        self.prompt = file_to_string(top_level_path+'prompts/vlm_non_agent_system_prompt.txt')

    def set_memory(self, memory: VideoMemory):
        self.memory = memory
        pass

    def query(self, question: str) -> AgentOutput:

        system_message = self.prompt
        memory_human_messages = construct_messages_from_memory(self.memory.get_working_memory())


        question_message = "Be sure to respond in the following format: \n\n"

        question_message += """```json
        {{
            "type_reasoning": "-input your reasoning in here for the type of question-", 
            "type": "-input the type of answer that is expected based only on the question: position, binary, time, or text. Be sure to then fill in that selected category.",
            "answer_reasoning", "-input your reasoning in here for the answer. If you do not know the answer, provide your best guess for the answer type you provide.-", 
            "text": "--a text answer here--",
            "binary": "yes/no",
            "position": "[x,y,z]",
            "orientation": "[-.92]", 
            "time": "5.3",
            "duration": "2.4",
        }}
        ```
        """
        question_message += f"Given the context above, please answer the question: {question}\n\n Follow the correct output format."


        inputs = [
            SystemMessage(content=system_message),
            memory_human_messages, 
            HumanMessage(content=question_message)
        ]

        while True:

            response = self.chain.invoke(inputs)
            response = ''.join(response.content.splitlines())
            print(response)
            try:
                if '```json' not in response:
                    # try parsing on its own since we cannot always trust llms
                    parsed = json.loads(response) 
                else:
                    parsed = parse_json(response)

                # then check it has all the required keys
                keys_to_check_for = ["time", "text", "binary", "position"]
                for key in keys_to_check_for:
                    if key not in parsed:
                        raise ValueError("Missing all the required keys during generate. Retrying...")
                response = AgentOutput.from_dict(parsed)

            except Exception as e:
                print(response)
                print(e)
                print("Generate call failed. Retrying...")
                continue

            break


        return response