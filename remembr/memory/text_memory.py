from dataclasses import dataclass, asdict

import datetime, time
from time import strftime, localtime
from typing import Any, Iterable, List, Optional, Tuple, Union
import numpy as np


from remembr.memory.memory import Memory, MemoryItem
from remembr.memory.memory_policy import MemoryPolicy, MemoryRecord, PolicyResult

# Due to Milvus DB's vector quantization, must normalize all times
FIXED_SUBTRACT=1721761000 # this is just a large value that brings us closed to 1970

class TextMemory(Memory):


    def __init__(self):
        self.memory = []

    def insert(self, item: MemoryItem, text_embedding=None):
        self.memory.append(item)

    def reset(self):
        self.memory = []

    def apply_policy(self, policy: MemoryPolicy, now: float = None) -> PolicyResult:
        # TextMemory stores no ids or embeddings, so records are identified by
        # their list index and the policy falls back to text similarity.
        records = [MemoryRecord(id=str(i), item=item) for i, item in enumerate(self.memory)]
        result = policy.select_for_removal(records, now=now)
        drop_indices = set(result.drop_ids)
        self.memory = [item for i, item in enumerate(self.memory) if str(i) not in drop_indices]
        return result

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
