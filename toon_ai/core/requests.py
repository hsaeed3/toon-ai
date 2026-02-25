"""toon_ai.core.requests"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Type,
    Generic,
    TypeVar,
    TYPE_CHECKING,
    Self,
)

from pydantic import BaseModel
from instructor.utils.core import prepare_response_model
from instructor.dsl.partial import Partial, MakeFieldsOptional

from .exceptions import ToonAITypeError

if TYPE_CHECKING:
    from instructor.models import KnownModelName
    from openai.types.chat import ChatCompletionMessageParam


OutputType = TypeVar("OutputType")


@dataclass
class ToonClientRequestParams(Generic[OutputType]):
    """
    Helper class for managing and normalizing various parameters used when invoking
    an LLM to generate a structured output using the TOON format.
    """

    model: KnownModelName | str
    """The identifier of the model to use.
    
    This should be in the LiteLLM format, e.g. `openai/gpt-4o-mini`.
    """

    messages: List[Dict[str, Any] | ChatCompletionMessageParam]
    """The messages to send to the LLM, in the OpenAI Chat Completions
    specificaiton."""

    output_type: Type[OutputType]
    """The type the model's response should be parsed into.
    
    This can be any Python primitive such as `int`, `float`, `bool`, or
    a Pydantic Model/TypedDict/etc.
    """

    base_url: str | None = None
    """An optional base URL to use for the API endpoint."""

    api_key: str | None = None
    """An optional/explicit API key to use for the API endpoint."""

    stream: bool = False
    """Whether to stream the response from the LLM."""

    kwargs: Dict[str, Any] = field(default_factory=dict)
    """Model-specific keyword arguments to pass to the API endpoint,
    e.g. `temperature`, `max_tokens`, `top_p`, `frequency_penalty`,"""

    @classmethod
    def prepare(
        cls,
        messages: str
        | Dict[str, Any]
        | List[Dict[str, Any] | ChatCompletionMessageParam],
        output_type: Type[OutputType] | Type,
        model: str = "openai/gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Prepare a ToonClientRequestParams instance for a given set of request parameters.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]

        return cls(
            model=model,
            messages=messages,
            output_type=output_type,
            base_url=base_url,
            api_key=api_key,
            kwargs=kwargs,
        )

    @property
    def normalized_output_type(self) -> Type[BaseModel | OutputType] | None:
        """
        Returns a normalized Pydantic model representation of the
        `output_type` specified in the request parameters.
        """
        if hasattr(self, "_normalized_output_type"):
            return self._normalized_output_type

        try:
            model = prepare_response_model(self.output_type)

            if not model:
                raise ToonAITypeError(
                    f"Failed to prepare response model for output type: {self.output_type}",
                )

            self._normalized_output_type = model
            return model
        except Exception as e:
            raise ToonAITypeError(
                f"Failed to prepare response model for output type: {self.output_type}",
            ) from e

    @property
    def partial_normalized_output_type(
        self,
    ) -> Type[BaseModel | OutputType] | None:
        """
        Returns a partial (all fields set to optional) representation of the
        output type specified in the request parameters.
        """
        if hasattr(self, "_partial_normalized_output_type"):
            return self._partial_normalized_output_type

        try:
            # Create a Partial[...] wrapper around the normalized model type
            partial_model = Partial[
                self.normalized_output_type, MakeFieldsOptional # type: ignore[type-abstract]
            ]
            get_partial = getattr(partial_model, "get_partial_model", None)

            if callable(get_partial):
                model = get_partial()
                self._partial_normalized_output_type = model
                return model

            self._partial_normalized_output_type = self.normalized_output_type
            return self.normalized_output_type
        except Exception as e:
            raise ToonAITypeError(
                f"Failed to prepare partial response model for output type: {self.output_type}",
            ) from e

    def inject_system_prompt(
        self,
        system_prompt: str,
    ) -> List[Dict[str, Any] | ChatCompletionMessageParam]:
        """
        Returns a mutated representation of the messages present within this
        `ToonClientRequestParams` instance, after the injection of system prompt
        content.
        """
        messages = self.messages.copy()

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

    def copy(self) -> Self:
        """
        Returns a copy of the request parameters.
        """
        return self.__class__(
            model=self.model,
            messages=self.messages.copy()
            if isinstance(self.messages, list)
            else self.messages,
            output_type=self.output_type,
            base_url=self.base_url if self.base_url else None,
            api_key=self.api_key if self.api_key else None,
            stream=self.stream,
            kwargs=self.kwargs.copy()
            if isinstance(self.kwargs, dict)
            else self.kwargs,
        )

    def dump(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the request parameters.
        """
        base_params: Dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }

        # Filter out None values without mutating during iteration
        cleaned_params = {
            k: v for k, v in base_params.items() if v is not None
        }

        # Merge in non-None kwargs
        for k, v in self.kwargs.items():
            if v is not None:
                cleaned_params[k] = v

        return cleaned_params
