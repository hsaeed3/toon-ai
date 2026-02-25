"""toon_ai.main"""

from __future__ import annotations

from typing import (
    Any,
    Iterable,
    Dict,
    List,
    Type,
    Literal,
    overload,
    TYPE_CHECKING,
)

from .core.client import (
    ToonClient,
    OutputType,
    OutputStrategy,
    RequestedType,
    LoggerVerbosity,
)

if TYPE_CHECKING:
    from tenacity import AsyncRetrying
    from instructor.models import KnownModelName
    from openai.types.chat import ChatCompletionMessageParam


@overload
def generate(
    messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
    output_type: None = None,
    model: str = "openai/gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    max_retries: int | AsyncRetrying = 3,
    verbosity: LoggerVerbosity | None = None,
    stream: Literal[False] = False,
    **kwargs: Any,
) -> OutputType: ...

@overload
def generate(
    messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
    output_type: Type[RequestedType],
    model: str = "openai/gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    max_retries: int | AsyncRetrying = 3,
    verbosity: LoggerVerbosity | None = None,
    stream: Literal[False] = False,
    **kwargs: Any,
) -> RequestedType: ...

@overload
def generate(
    messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
    output_type: None = None,
    model: str = "openai/gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    max_retries: int | AsyncRetrying = 3,
    verbosity: LoggerVerbosity | None = None,
    stream: Literal[True] = True,
    **kwargs: Any,
) -> Iterable[OutputType]: ...

@overload
def generate(
    messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
    output_type: Type[RequestedType],
    model: str = "openai/gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    max_retries: int | AsyncRetrying = 3,
    verbosity: LoggerVerbosity | None = None,
    stream: Literal[True] = True,
    **kwargs: Any,
) -> Iterable[RequestedType]: ...

def generate(
    messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
    output_type: Type[Any] | None = None,
    strategy: OutputStrategy = "text",
    model: KnownModelName | str = "openai/gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    max_retries: int | AsyncRetrying = 3,
    verbosity: LoggerVerbosity | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> OutputType | Iterable[OutputType]:
    """
    Generate a structured output from an LLM using the TOON format for
    response generation and parsing, according to the strategy set
    on this client instance.

    Args:
        messages: The messages to send to the LLM, in the OpenAI Chat Completions
            specificaiton.
        output_type: The type the model's response should be parsed into.
            This can be any Python primitive such as `int`, `float`, `bool`, or
            a Pydantic Model/TypedDict/etc.
        strategy: The strategy to use for generating the structured output. Defaults to `text`.
        model: The model to use for generating the structured output. Defaults to `openai/gpt-4o-mini`.
        base_url: An optional base URL to use for the API endpoint.
        api_key: An optional/explicit API key to use for the API endpoint.
        max_retries: The maximum number of retries to attempt if the request fails.
        verbosity: The verbosity of the logger.
        stream: Whether to stream the response from the LLM.
        **kwargs: Additional keyword arguments to pass to the LLM.

    Returns:
        OutputType | AsyncIterable[OutputType]
            The structured output from the LLM, or an async iterable of structured outputs
            if streaming is enabled.
    """
    client = ToonClient(
        strategy=strategy,
    )

    return client.generate(
        messages=messages,
        output_type=output_type,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_retries=max_retries,
        verbosity=verbosity,
        stream=stream,
        **kwargs,
    )