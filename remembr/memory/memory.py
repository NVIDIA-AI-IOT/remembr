from dataclasses import dataclass
import inspect 

@dataclass
class MemoryItem:
    caption: str
    time: float
    position: list
    theta: float
    # Which camera produced the caption ('' when unknown / single-camera).
    # In multi-camera setups, theta is the viewing direction of this camera
    # (robot yaw + camera mounting offset), not the robot's base heading.
    camera_id: str = ''

    @classmethod
    def from_dict(cls, dict_input):      
        return cls(**{
            k: v for k, v in dict_input.items() 
            if k in inspect.signature(cls).parameters
        })
    
    def __post_init__(self):
        # Not every method will use a caption, so we set it to none in those cases
        if self.caption is None:
            self.caption = ''
        if self.camera_id is None:
            self.camera_id = ''


class Memory:

    def insert(self, item: MemoryItem):
        raise NotImplementedError

    def remove(self, ids: list):
        raise NotImplementedError

    def apply_policy(self, policy, now: float = None):
        # policy is a remembr.memory.memory_policy.MemoryPolicy; implementations
        # should run it over their stored records, delete what it selects, and
        # return the resulting PolicyResult.
        raise NotImplementedError

    def get_working_memory(self) -> list[MemoryItem]:
        raise NotImplementedError

    def search_by_position(self, query: tuple) -> list[MemoryItem]:
        raise NotImplementedError

    def search_by_time(self, hms_time_query: str) -> list[MemoryItem]:
        raise NotImplementedError

    def search_by_text(self, query: str) -> list[MemoryItem]:
        raise NotImplementedError

    def memory_to_string(self, memory_list: list[MemoryItem]) -> str:
        raise NotImplementedError

