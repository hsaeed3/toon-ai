"""toon_ai.strategies.abstract"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Literal,
    Generic,
    TypeAlias,
    TypeVar,
    TYPE_CHECKING,
)

from ..core.requests import ToonClientRequestParams

if TYPE_CHECKING:
    from litellm import ModelResponse


OutputType = TypeVar("OutputType")


OutputStrategy: TypeAlias = Literal["text", "tools"]
"""The strategy to use for prompting/structuring the model's request when
generating TOON structured outputs."""


@dataclass
class AbstractStrategy(ABC, Generic[OutputType]):
    """
    Abstract base class for a strategy, which provides the ability
    to format a set of request parameters in the OpenAI Chat Completions
    format into a representation in which a model's response can then
    be parsed to result in a structured output.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of this strategy.
        """
        pass

    @abstractmethod
    def format_request_params(
        self,
        request_params: ToonClientRequestParams[OutputType],
    ) -> ToonClientRequestParams[OutputType]:
        """
        Mutates a ToonClientRequestParams instance to prompt/structure the
        model's request parameters in the context of the strategy, and returns
        a new ToonClientRequestParams instance.
        """

    @abstractmethod
    def format_retry_params(
        self,
        request_params: ToonClientRequestParams[OutputType],
        exception: Exception,
        last_response: Any | None,
    ) -> ToonClientRequestParams[OutputType]:
        """
        Mutates a ToonClientRequestParams instance to prompt/structure the
        model's request parameters in the context of the strategy, and returns
        a new ToonClientRequestParams instance.
        """

    @abstractmethod
    def parse_response(
        self,
        request_params: ToonClientRequestParams[OutputType],
        response: ModelResponse,
    ) -> OutputType:
        """
        Parses a litellm.ModelResponse instance to result in a structured output,
        using the `output_type` specified in the ModelRequestParams instance.
        """

    @abstractmethod
    async def async_parse_response_chunk(
        self,
        request_params: ToonClientRequestParams[OutputType],
        toon_chunks: AsyncGenerator[str, None],
    ) -> AsyncGenerator[OutputType, None]:
        """
        Parses a TOON streaming response chunk to result in a structured output,
        using the `output_type` specified in the ToonClientRequestParams instance.
        """
