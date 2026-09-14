import json


class Budget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tool_calls = 0
        self.total_frames = 0
        self.seen = set()

    def can_call(self) -> bool:
        return self.tool_calls < self.cfg.max_tool_calls

    def signature(self, name, arguments) -> str:
        return name + ":" + json.dumps(arguments, sort_keys=True, default=str)

    def seen_before(self, name, arguments) -> bool:
        return self.signature(name, arguments) in self.seen

    def register(self, name, arguments, obs) -> None:
        self.tool_calls += 1
        self.seen.add(self.signature(name, arguments))
        self.total_frames += obs.n_frames

    def frames_left(self) -> int:
        return max(0, self.cfg.max_total_frames - self.total_frames)
