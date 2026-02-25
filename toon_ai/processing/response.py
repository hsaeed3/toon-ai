"""toon_ai.processing.response"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
import re
from typing import TYPE_CHECKING

from ..core.logger import _log_debug_context

if TYPE_CHECKING:
    from litellm import ModelResponse


_logger = logging.getLogger("toon_ai.processing.response")


def parse_toon_block_from_model_response(
    response: ModelResponse,
) -> str | None:
    """
    Extracts a TOON code block (if any was found) from a LiteLLM
    ModelResponse.

    If no TOON code block is found, returns None.

    Args:
        response : ModelResponse
            The LiteLLM ModelResponse to extract a TOON code block from.

    Returns:
        str | None
            The TOON code block if found, otherwise None.
    """
    if not response.choices:
        return None
    if not response.choices[0].message:  # type: ignore[union-attr]
        return None
    if not response.choices[0].message.content or not isinstance( # type: ignore
        response.choices[0].message.content, str # type: ignore
    ):
        return None

    text = response.choices[0].message.content  # type: ignore[union-attr]
    if not text or not isinstance(text, str):
        return None

    toon_match = re.search(
        r"```toon\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if toon_match:
        content = toon_match.group(1).strip()
        _log_debug_context(
            logger=_logger,
            lines=[
                f"Found TOON code block: [italic]{content}[/italic].",
            ],
        )
        return content

    generic_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if generic_match:
        content = generic_match.group(1).strip()
        _log_debug_context(
            logger=_logger,
            lines=[
                f"Found generic code block: [italic]{content}[/italic].",
            ],
        )
        return content

    return None


async def async_parse_code_block_from_stream(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """
    Extracts content from the first code block in an async stream of chunks.

    Extracts content from ```lang ... ``` or generic ``` ... ``` blocks.
    Yields characters inside the code block as they arrive.

    NOTE: This function only extracts from the first code block;
    and ignores any subsequent blocks.
    """
    in_codeblock = False
    backtick_count = 0
    skip_lang_tag = False
    done = False

    async for chunk in chunks:
        if done:
            break
        for char in chunk:
            if not in_codeblock:
                if char == "`":
                    backtick_count += 1
                    if backtick_count == 3:
                        in_codeblock = True
                        backtick_count = 0
                        skip_lang_tag = True
                else:
                    backtick_count = 0
            else:
                if skip_lang_tag:
                    if char == "\n":
                        skip_lang_tag = False
                    continue

                if char == "`":
                    backtick_count += 1
                    if backtick_count == 3:
                        done = True
                        break
                else:
                    if backtick_count > 0:
                        for _ in range(backtick_count):
                            yield "`"
                        backtick_count = 0
                    yield char
