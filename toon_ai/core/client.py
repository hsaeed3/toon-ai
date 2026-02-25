"""toon_ai.core.client

Provides the `ToonClient` class, which is a wrapper around a specific
model provider SDK's client such as OpenAI, Anthropic, LiteLLM, etc. and
provides a unified interface for using the provider's API to generate
structured outputs, while prompting and parsing responses in the
TOON format.
"""

from __future__ import annotations

import logging
import asyncio
import inspect
from dataclasses import dataclass, field
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Generator,
    Iterable,
)
from typing import (
    final,
    Any,
    Awaitable,
    Dict,
    Generic,
    List,
    Type,
    TypeVar,
    overload,
    Literal,
    TYPE_CHECKING,
)

from instructor.core.retry import initialize_retrying
from tenacity import AsyncRetrying, RetryError

from ..processing.response import async_parse_code_block_from_stream
from ..strategies.abstract import AbstractStrategy, OutputStrategy
from ..strategies.text import TextStrategy
from .requests import ToonClientRequestParams
from .logger import (
    set_logger_verbosity,
    LoggerVerbosity,
    _log_info_panel,
    _log_debug_context,
)
from .exceptions import (
    ToonAIRequestError,
    ToonAIResponseError,
    ToonAITypeError,
)

if TYPE_CHECKING:
    from instructor.models import KnownModelName
    from openai.types.chat import ChatCompletionMessageParam


_logger = logging.getLogger("toon_ai.core.client")


OutputType = TypeVar("OutputType")
RequestedType = TypeVar("RequestedType")


_TOON_AI_LITELLM_INSTANCE = None


def _get_litellm():
    global _TOON_AI_LITELLM_INSTANCE
    if _TOON_AI_LITELLM_INSTANCE is not None:
        return _TOON_AI_LITELLM_INSTANCE

    import litellm

    litellm.drop_params = True
    litellm.modify_params = True

    _TOON_AI_LITELLM_INSTANCE = litellm
    return _TOON_AI_LITELLM_INSTANCE


def _extract_async_iterable(value: Any) -> AsyncIterable[OutputType]:
    if hasattr(value, "__aiter__"):
        return value
    if isinstance(value, tuple):
        for item in value:
            if hasattr(item, "__aiter__"):
                return item
    raise ToonAITypeError(
        f"Expected an async iterable, got {type(value).__name__}.",
    )


def _run_async_iterable(
    iterable_coro: Awaitable[AsyncIterable[OutputType]]
    | AsyncIterable[OutputType],
) -> Generator[OutputType, None, None]:
    loop = asyncio.new_event_loop()
    async_generator = None
    try:
        asyncio.set_event_loop(loop)
        if inspect.isawaitable(iterable_coro):
            result = loop.run_until_complete(iterable_coro)
        else:
            result = iterable_coro
        async_iterable = _extract_async_iterable(result)
        async_generator = async_iterable.__aiter__()
        while True:
            try:
                item = loop.run_until_complete(async_generator.__anext__())
            except StopAsyncIteration:
                break
            yield item
    finally:
        if async_generator is not None:
            try:
                loop.run_until_complete(async_generator.aclose())  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            pass
        loop.close()


@final
@dataclass(init=False)
class ToonClient(Generic[OutputType]):
    """
    A wrapper around the LiteLLM SDK, for generating structured outputs
    with LLMs in the TOON format.
    """

    output_type: Type[OutputType] | None = None
    """The default output type of which structured outputs should be generated in,
    if one is not given when invoking the `generate` method."""

    _strategy: OutputStrategy = field(default="text")
    _strategy_instance: AbstractStrategy[OutputType] | None = field(
        default=None
    )

    def _set_strategy_instance(self, strategy: OutputStrategy) -> None:
        if strategy == "text":
            self._strategy_instance = TextStrategy()
            self._strategy = "text"
        elif strategy == "tools":
            raise NotImplementedError("Tools strategy is not yet implemented.")
        else:
            raise ValueError(f"Invalid output strategy: {strategy}")

    @property
    def strategy(self) -> OutputStrategy:
        """
        The strategy to use when generating structured outputs using the `generate` method.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: OutputStrategy) -> None:
        """
        Set the strategy to use when generating structured outputs using the `generate` method.
        """
        self._set_strategy_instance(strategy)

    def __init__(
        self,
        output_type: Type[OutputType] | None = None,
        strategy: OutputStrategy = "text",
    ) -> None:
        """
        Initialize a ToonClient instance with an optionally set default model,
        output type, and strategy.

        Args:
            output_type: The default output type of which structured outputs should be generated in,
                if one is not given when invoking the `generate` method.
            strategy: The default/set strategy to use when generating structured outputs
                using the `generate` method.
        """
        self.output_type = output_type
        self._set_strategy_instance(strategy)

    @overload
    async def async_generate(
        self,
        messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
        output_type: None = None,
        model: KnownModelName | str = "openai/gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int | AsyncRetrying = 3,
        verbosity: LoggerVerbosity | None = None,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> OutputType: ...

    @overload
    async def async_generate(
        self,
        messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
        output_type: Type[RequestedType],
        model: KnownModelName | str = "openai/gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int | AsyncRetrying = 3,
        verbosity: LoggerVerbosity | None = None,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> RequestedType: ...

    @overload
    async def async_generate(
        self,
        messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
        output_type: None = None,
        model: KnownModelName | str = "openai/gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int | AsyncRetrying = 3,
        verbosity: LoggerVerbosity | None = None,
        stream: Literal[True] = True,
        **kwargs: Any,
    ) -> AsyncIterable[OutputType]: ...

    @overload
    async def async_generate(
        self,
        messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
        output_type: Type[RequestedType],
        model: KnownModelName | str = "openai/gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int | AsyncRetrying = 3,
        verbosity: LoggerVerbosity | None = None,
        stream: Literal[True] = True,
        **kwargs: Any,
    ) -> AsyncIterable[RequestedType]: ...

    async def async_generate(
        self,
        messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
        output_type: Type[Any] | None = None,
        model: KnownModelName | str = "openai/gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int | AsyncRetrying = 3,
        verbosity: LoggerVerbosity | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> OutputType | AsyncIterable[OutputType]:
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
        if verbosity:
            set_logger_verbosity(verbosity)

        base_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
            **kwargs,
        }
        resolved_output_type = (
            output_type
            if output_type is not None
            else self.output_type
            if self.output_type is not None
            else str
        )
        base_params = ToonClientRequestParams.prepare(
            messages=messages,
            output_type=resolved_output_type,
            model=model,
            **base_kwargs,
        )

        _log_info_panel(
            logger=_logger,
            title=f"Preparing Structured Output For Type: [bold]{resolved_output_type}[/bold]",
            lines=[
                f"Model: [italic]{model}[/italic]",
                f"Streaming: [italic]{stream}[/italic]",
                f"Strategy: [italic]{self.strategy}[/italic]",
                f"Max Retries: [italic]{max_retries}[/italic]",
            ],
        )
        _log_debug_context(
            logger=_logger,
            lines=[
                f"Recieved `messages` of length: [italic]{len(base_params.messages)}[/italic].",
                f"Ingested `output_type` of type: [italic]{resolved_output_type}[/italic].",
                f"Normalized `output_type` to: [italic]{base_params.normalized_output_type}[/italic].",
            ],
        )

        strategy = self._strategy_instance
        if strategy is None:
            raise ToonAIRequestError(
                "No output strategy instance is configured. Please set a strategy using the `strategy` property.",
            )

        # ------------------------------------------------------------- #
        # STREAMING
        # ------------------------------------------------------------- #

        if stream:
            formatted_params = strategy.format_request_params(base_params)
            _log_debug_context(
                logger=_logger,
                lines=[
                    f"Formatted request parameters for output strategy: [italic]{strategy.name}[/italic].",
                    f"New `messages` length: [italic]{len(formatted_params.messages)}[/italic].",
                ],
            )

            _log_info_panel(
                logger=_logger,
                title="Generating Structured Output in Streaming Mode",
                lines=[
                    f"Strategy: [italic]{strategy.name}[/italic]",
                ],
            )

            async def stream_generator() -> AsyncGenerator[OutputType, None]:
                response_stream = None
                toon_chunks = None
                content_stream = None
                try:
                    _log_debug_context(
                        logger=_logger,
                        lines=[
                            f"Invoking LiteLLM to stream a response with parameters: [italic]{formatted_params.dump()}[/italic].",
                        ],
                    )

                    response_stream = await _get_litellm().acompletion(
                        **formatted_params.dump(), stream=True
                    )

                    async def content_chunks() -> AsyncGenerator[str, None]:
                        try:
                            async for chunk in response_stream:
                                if (
                                    not chunk
                                    or not hasattr(chunk, "choices")
                                    or not chunk.choices
                                ):
                                    continue
                                choice = chunk.choices[0]
                                # Streaming providers typically send delta content.
                                if hasattr(choice, "delta") and choice.delta:
                                    if (
                                        hasattr(choice.delta, "content")
                                        and choice.delta.content
                                    ):
                                        yield choice.delta.content
                                        continue
                                # Fallback for providers that send full message content.
                                if (
                                    hasattr(choice, "message")
                                    and choice.message
                                ):
                                    if (
                                        hasattr(choice.message, "content")
                                        and choice.message.content
                                    ):
                                        yield choice.message.content
                        finally:
                            # Ensure the underlying async generator is properly closed
                            if hasattr(response_stream, "aclose"):
                                try:
                                    await response_stream.aclose()
                                except Exception:
                                    pass

                    content_stream = content_chunks()
                    toon_chunks = async_parse_code_block_from_stream(
                        content_stream
                    )
                    try:
                        async for item in strategy.async_parse_response_chunk(
                            request_params=formatted_params,
                            toon_chunks=toon_chunks,
                        ):  # type: ignore[attr-defined]
                            yield item
                    finally:
                        if toon_chunks is not None and hasattr(
                            toon_chunks, "aclose"
                        ):
                            try:
                                await toon_chunks.aclose()
                            except Exception:
                                pass
                        if content_stream is not None and hasattr(
                            content_stream, "aclose"
                        ):
                            try:
                                await content_stream.aclose()
                            except Exception:
                                pass
                except Exception as e:
                    raise ToonAIRequestError(
                        f"Failed to invoke LiteLLM to stream a response: {e}",
                    ) from e
                finally:
                    if response_stream is not None and hasattr(
                        response_stream, "aclose"
                    ):
                        try:
                            await response_stream.aclose()
                        except Exception:
                            pass

            return stream_generator()

        timeout = kwargs.get("timeout")
        retrying = initialize_retrying(
            max_retries=max_retries, is_async=True, timeout=timeout
        )

        # ------------------------------------------------------------- #
        # NON STREAMING
        #
        # NOTE: Currently ive only implemented retries for non-streamed responses,
        # not sure if i will do it for streamed responses as well.
        # ------------------------------------------------------------- #

        last_exception: Exception | None = None
        last_response: Any | None = None

        try:
            async for retry_attempt in retrying:
                with retry_attempt:
                    attempt_number = getattr(
                        retry_attempt, "attempt_number", 1
                    )

                    # First attempt uses the normal formatted params; subsequent
                    # attempts use the strategy's retry formatting.
                    if attempt_number == 1:
                        formatted_params = strategy.format_request_params(
                            base_params
                        )
                    else:
                        formatted_params = strategy.format_retry_params(
                            base_params,
                            exception=last_exception
                            if last_exception is not None
                            else ToonAIResponseError(
                                "Previous attempt failed."
                            ),
                            last_response=last_response,
                        )

                    _log_debug_context(
                        logger=_logger,
                        lines=[
                            f"Formatted request parameters for retry attempt: [italic]{attempt_number}[/italic] output strategy: [italic]{strategy.name}[/italic].",
                            f"New `messages` length: [italic]{len(formatted_params.messages)}[/italic].",
                        ],
                    )

                    try:
                        response = await _get_litellm().acompletion(
                            **formatted_params.dump(), stream=False
                        )
                    except Exception as e:
                        last_exception = e
                        raise ToonAIRequestError(
                            f"Failed to invoke LiteLLM to generate a response during retry attempt: {e}",
                        ) from e

                    if not response:
                        last_exception = ToonAIResponseError(
                            "No response received from the model provider.",
                        )
                        raise last_exception

                    try:
                        return strategy.parse_response(
                            request_params=formatted_params, response=response
                        )
                    except Exception as e:
                        last_exception = e
                        last_response = response

                        if (
                            isinstance(max_retries, int)
                            and attempt_number < max_retries
                        ):
                            _log_info_panel(
                                logger=_logger,
                                title="Encountered an Error Parsing the Response Retrying...",
                                lines=[
                                    f"Retry Attempt: [italic]{attempt_number}[/italic]",
                                    f"Max Retries: [italic]{max_retries}[/italic]",
                                ],
                            )
                            _log_debug_context(
                                logger=_logger,
                                lines=[
                                    f"Raw response content: [italic]{response if response else 'No content'}[/italic].",
                                ],
                            )

                        raise ToonAIResponseError(
                            f"Failed to parse provider's response into TOON format: {e}",
                            raw_response=response,
                        )
        except RetryError as e:
            raise ToonAIRequestError(
                f"Failed after {e.last_attempt.attempt_number} attempts: {e}",
            ) from e

        raise ToonAIResponseError(
            "Failed to generate a response.",
        )

    @overload
    def generate(
        self,
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
        self,
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
        self,
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
        self,
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
        self,
        messages: str | List[Dict[str, Any] | ChatCompletionMessageParam],
        output_type: Type[Any] | None = None,
        model: str = "openai/gpt-4o-mini",
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
        if output_type is None:
            output_type = self.output_type or str

        if stream:
            async_iterable = (
                asyncio.run(
                    self.async_generate(
                        messages=messages,
                        output_type=output_type,
                        model=model,
                        base_url=base_url,
                        api_key=api_key,
                        max_retries=max_retries,
                        verbosity=verbosity,
                        stream=True,
                        **kwargs,
                    ),
                ),
            )
            return _run_async_iterable(asyncio.sleep(0, result=async_iterable))

        result = asyncio.run(
            self.async_generate(
                messages=messages,
                output_type=output_type,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_retries=max_retries,
                verbosity=verbosity,
                stream=False,
                **kwargs,
            ),
        )
        if isinstance(result, AsyncIterable):
            return _run_async_iterable(asyncio.sleep(0, result=result))  # type: ignore[return-value]
        return result  # type: ignore[return-value]
