from dataclasses import dataclass, asdict

import datetime, time
from time import strftime, localtime
from typing import Any, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document
import numpy as np


from remembr.memory.memory import Memory, MemoryItem
from remembr.captioners.captioner import Captioner

from langchain_community.vectorstores import Milvus

# Due to Milvus DB's vector quantization, must normalize all times
FIXED_SUBTRACT=1721761000 # this is just a large value that brings us closed to 1970

class TextMemory(Memory):


    def __init__(self):
        self.memory = []

    def insert(self, item: MemoryItem, text_embedding=None):
        self.memory.append(item)

    def reset(self):
        self.memory = []

    def get_working_memory(self) -> list[MemoryItem]:
        if type(self.memory[0]) == str:
            # if already a string then return the string itself appended as a list
            return "\n".join(self.memory)
        return self.memory



  

    def memory_to_string(self, memory_item_list: list[MemoryItem]) -> str:
        out_string = ""
        for doc in memory_item_list:
            t = doc.time
            
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            s = f"At time={t}, the robot was at an average position of {np.array(doc.position).round(3).tolist()} with an average orientation of {doc.theta} radians. "
            s += f"The robot saw the following: {doc.caption}\n\n"
            out_string += s
        return out_string
