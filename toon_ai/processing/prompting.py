"""toon_ai.processing.prompting"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


def inject_system_prompt(
    system_prompt: str,
    messages: List[Dict[str, Any] | ChatCompletionMessageParam],
) -> List[Dict[str, Any] | ChatCompletionMessageParam]:
    """
    Injects a system prompt into a list of messages.
    """
    if not len(messages) > 0:
        # assume messages arent negative
        return [
            {"role": "system", "content": system_prompt},
        ]

    if "role" in messages[0] and messages[0]["role"] == "system":
        user_system_prompt = messages.pop(0)["content"]
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"{user_system_prompt}\n\n{system_prompt}",
            },
        )
        return messages

    messages.insert(
        0,
        {"role": "system", "content": system_prompt},
    )
    return messages
