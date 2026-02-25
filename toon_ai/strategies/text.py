"""toon_ai.strategies.text"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import (
    Any,
    AsyncGenerator,
    Type,
    TYPE_CHECKING,
)

from pydantic import BaseModel
from toon_format import decode

from instructor.dsl.partial import Partial, MakeFieldsOptional

from ..core.exceptions import ToonAIResponseError
from ..processing.toon import (
    _coerce_enums_for_model,
    _coerce_output_value,
    encode_toon_structure,
    decode_toon,
)
from ..processing.prompting import inject_system_prompt
from ..processing.response import parse_toon_block_from_model_response
from ..strategies.abstract import AbstractStrategy, OutputType
from ..core.requests import ToonClientRequestParams

if TYPE_CHECKING:
    from litellm import ModelResponse


def get_text_strategy_system_prompt(normalized_type: Type[BaseModel]) -> str:
    """
    Generate the TOON system prompt for a given response model using the
    `text` strategy.

    Args:
        response_model: A Pydantic BaseModel class

    Returns:
        A formatted system prompt string for TOON mode
    """
    toon_structure = encode_toon_structure(normalized_type)
    return dedent(f"""
        Respond in TOON format inside a code block starting and ending with ````toon`.

        Structure:
        ```toon
{toon_structure}
        ```

        Rules:
        - 2-space indentation for nesting
        - Arrays: field[N]: val1,val2,val3 where N = actual count
        - Tables: field[N,]{{col1,col2}}: with one row per line

        Value formatting:
        - <int>: whole numbers without quotes (e.g., age: 25)
        - <float>: decimal numbers without quotes (e.g., price: 19.99)
        - "<str>": quoted strings (e.g., name: "Alice", zip: "10001")
        - <bool>: true or false

        IMPORTANT: Output values only, not the type placeholders.
    """).strip()


def get_text_strategy_retry_message(
    exception: Exception,
    previous_response: str = "",
) -> str:
    """
    Generate a retry message for TOON validation errors using the
    `text` strategy.

    Args:
        exception: The validation exception that occurred
        previous_response: Optional previous response text for context

    Returns:
        A formatted reask message string
    """
    base_message = (
        f"Validation error:\n{exception}\n\n"
        "Fix your TOON response:\n"
        "- int fields: whole numbers, no quotes (age: 25)\n"
        "- float fields: decimals, no quotes (price: 19.99)\n"
        '- str fields: quoted (name: "Alice")\n'
        "- Array [N] must match actual count\n\n"
        "Return corrected TOON in code block starting and ending with ````toon`."
    )

    if previous_response:
        return (
            f"Validation error:\n{exception}\n\nYour previous response:\n{previous_response}\n\n"
            + base_message.split("\n\n", 1)[1]
        )

    return base_message


def partial_normalized_type(
    normalized_type: Type[BaseModel],
) -> Type[BaseModel]:
    partial_type: Any = Partial[normalized_type, MakeFieldsOptional]  # type: ignore[type-abstract]
    get_partial = getattr(partial_type, "get_partial_model", None)
    if callable(get_partial):
        return get_partial()
    return normalized_type


@dataclass
class TextStrategy(AbstractStrategy[OutputType]):
    """
    Structured output strategy that uses a method that prompts an LLM to generate
    a TOON formatted content as a ```toon``` code block of the input `output_type`.
    """

    @property
    def name(self) -> str:
        """
        The name of this strategy.
        """
        return "text"

    def format_request_params(
        self,
        request_params: ToonClientRequestParams[OutputType],
    ) -> ToonClientRequestParams[OutputType]:
        """
        Mutates a ModelRequestParams instance to prompt/structure the
        model's request parameters in the context of the strategy.
        """
        request_params = request_params.copy()

        system_prompt = get_text_strategy_system_prompt(
            request_params.normalized_output_type # type: ignore[arg-type]
        )

        request_params.messages = inject_system_prompt(
            system_prompt=system_prompt,
            messages=request_params.messages,
        )
        return request_params

    def format_retry_params(
        self,
        request_params: ToonClientRequestParams[OutputType],
        exception: Exception,
        last_response: str | None,
    ) -> ToonClientRequestParams[OutputType]:
        """
        Mutates a ModelRequestParams instance to prompt/structure the
        model's request parameters in the context of the strategy.
        """
        request_params = request_params.copy()

        if last_response and not isinstance(last_response, str):
            last_response = str(last_response)

        retry_message = get_text_strategy_retry_message(
            exception=exception,
            previous_response=last_response,  # type: ignore[arg-type]
        )
        request_params.messages = inject_system_prompt(
            system_prompt=retry_message,
            messages=request_params.messages,
        )

        return request_params

    def parse_response(
        self,
        request_params: ToonClientRequestParams[OutputType],
        response: ModelResponse,
    ) -> OutputType:
        """
        Parses a litellm.ModelResponse instance to result in a structured output,
        using the `output_type` specified in the ModelRequestParams instance.
        """
        parsed_toon = parse_toon_block_from_model_response(response)

        if not parsed_toon:
            raise ToonAIResponseError(
                "No TOON content found in response",
                raw_response=response,
            )

        decoded_toon = decode_toon(
            normalized_type=request_params.normalized_output_type,  # type: ignore[arg-type]
            toon_content=parsed_toon,
        )

        if isinstance(request_params.output_type, type) and not issubclass(
            request_params.output_type, BaseModel
        ):
            if hasattr(decoded_toon, "content"):
                return getattr(decoded_toon, "content")

        return decoded_toon  # type: ignore[return-value]

    async def async_parse_response_chunk(
        self,
        request_params: ToonClientRequestParams[OutputType],
        toon_chunks: AsyncGenerator[str, None],
    ) -> AsyncGenerator[OutputType, None]:  # type: ignore[override]
        """
        Parses a TOON streaming response chunk to result in a structured output,
        using the `output_type` specified in the ToonClientRequestParams instance.
        """
        base_model = request_params.normalized_output_type
        partial_model = partial_normalized_type(base_model)  # type: ignore[arg-type]
        full_content = ""
        last_successful_data: dict[str, Any] | None = None

        async for chunk in toon_chunks:
            full_content += chunk
            if "\n" not in full_content:
                continue

            last_newline = full_content.rfind("\n")
            complete_lines = full_content[: last_newline + 1]

            try:
                decoded = decode(complete_lines.strip())
                if (
                    isinstance(decoded, dict)
                    and decoded
                    and decoded != last_successful_data
                ):
                    last_successful_data = decoded
                    data = _coerce_enums_for_model(base_model, decoded)  # type: ignore[arg-type]
                    model = partial_model.model_validate(data, strict=None)
                    yield _coerce_output_value(
                        request_params=request_params,
                        partial_model=model,
                    )
            except Exception:
                pass

        if full_content.strip():
            try:
                decoded = decode(full_content.strip())
                if isinstance(decoded, dict):
                    data = _coerce_enums_for_model(base_model, decoded)  # type: ignore[arg-type]
                    model = partial_model.model_validate(data, strict=None)
                    yield _coerce_output_value(
                        request_params=request_params,
                        partial_model=model,
                    )
                    return
            except Exception:
                pass

            if last_successful_data:
                data = _coerce_enums_for_model(
                    base_model, last_successful_data # type: ignore[arg-type]
                ) 
                model = partial_model.model_validate(data, strict=None)
                yield _coerce_output_value(
                    request_params=request_params,
                    partial_model=model,
                )
