"""Echo's multi-image tool adapter for the upstream VeRL agent loop."""
from copy import deepcopy
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


def expand_tool_image_placeholders(messages, image_count):
    """Match image markers to one tool response's image list.

    The pinned upstream loop emits one marker per response even when its
    ToolResponse.image contains several images. It then passes every image to
    the processor. Repair that mismatch while preserving all images and text.
    This helper only handles the newly appended tool messages, not full history.
    """
    if not image_count or not messages or any(m.get('role') != 'tool' for m in messages):
        return messages
    locations = [(i, j) for i, message in enumerate(messages)
                 if isinstance(message.get('content'), list)
                 for j, item in enumerate(message['content']) if item.get('type') == 'image']
    if len(locations) == image_count:
        return messages
    if len(messages) != 1 or len(locations) != 1:
        raise ValueError('Cannot assign multiple returned images across ambiguous parallel tool responses')
    repaired = deepcopy(messages)
    i, j = locations[0]
    content = repaired[i]['content']
    content[j:j + 1] = [dict(content[j]) for _ in range(image_count)]
    return repaired


class EchoToolAgentLoop(ToolAgentLoop):
    async def apply_chat_template(self, messages, tools=None, images=None, videos=None,
                                  remove_system_prompt=False):
        if remove_system_prompt and images:
            messages = expand_tool_image_placeholders(messages, len(images))
        return await super().apply_chat_template(
            messages, tools=tools, images=images, videos=videos,
            remove_system_prompt=remove_system_prompt)
