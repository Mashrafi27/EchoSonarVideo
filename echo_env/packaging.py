_HEAD = "<|im_end|>\n<|im_start|>user\n"
_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
_TOOL_RESPONSE = "<tool_response>\n<image>\n</tool_response>\n"


def to_deepeyes_obs(observation, user_prompt: str) -> dict:
    if not observation.ok or observation.n_frames == 0:
        return {"prompt": _HEAD + f"Error: {observation.error or observation.text}" + _TAIL}
    tool_response = _TOOL_RESPONSE * observation.n_frames
    return {
        "prompt": _HEAD + tool_response + user_prompt + _TAIL,
        "multi_modal_data": {"image": [f.image for f in observation.frames]},
    }
