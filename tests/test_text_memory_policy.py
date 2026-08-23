from remembr.memory.memory import MemoryItem
from remembr.memory.memory_policy import StaleDuplicatePolicy
from remembr.memory.text_memory import TextMemory

T0 = 1_722_000_000.0


def item(caption, t, position=(0.0, 0.0, 0.0)):
    return MemoryItem(caption=caption, time=t, position=list(position), theta=0.0)


def test_apply_policy_prunes_stale_duplicates():
    memory = TextMemory()
    for i in range(5):
        memory.insert(item('i see a desk', T0 + i))
    memory.insert(item('a person walks past the desk', T0 + 2, position=(0.2, 0.0, 0.0)))
    memory.insert(item('i see a desk', T0 + 10_000))

    policy = StaleDuplicatePolicy(max_age=600.0, position_radius=1.0)
    result = memory.apply_policy(policy)

    assert result.num_scanned == 7
    assert result.num_dropped == 5

    remaining = memory.get_working_memory()
    captions_and_times = [(m.caption, m.time) for m in remaining]
    assert ('a person walks past the desk', T0 + 2) in captions_and_times
    assert ('i see a desk', T0 + 10_000) in captions_and_times
    assert len(remaining) == 2


def test_apply_policy_noop_on_fresh_memory():
    memory = TextMemory()
    for i in range(3):
        memory.insert(item('i see a desk', T0 + i))

    policy = StaleDuplicatePolicy(max_age=600.0, position_radius=1.0)
    result = memory.apply_policy(policy)

    assert result.num_dropped == 0
    assert len(memory.get_working_memory()) == 3


def test_apply_policy_repeated_application_is_stable():
    memory = TextMemory()
    for i in range(4):
        memory.insert(item('i see a desk', T0 + i))
    memory.insert(item('i see a desk', T0 + 10_000))

    policy = StaleDuplicatePolicy(max_age=600.0, position_radius=1.0)
    first = memory.apply_policy(policy)
    second = memory.apply_policy(policy)

    assert first.num_dropped == 4
    assert second.num_dropped == 0
    assert len(memory.get_working_memory()) == 1
