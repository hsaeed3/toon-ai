"""toon_ai.core.exceptions"""

from typing import Any

from rich.traceback import install

install()

__all__ = (
    "ToonAIError",
    "ToonAIModelError",
    "ToonAITypeError",
    "ToonAIRequestError",
    "ToonAIResponseError",
)


class ToonAIError(Exception):
    """
    Base exception class for all custom errors raised by `toon-ai`.
    """

    pass


class ToonAIModelError(ToonAIError):
    """
    Exception raised if an issue happens to occur during the initial
    configuration of a `Model` using LiteLLM.
    """

    pass


class ToonAITypeError(ToonAIError, TypeError):
    """
    Exception raised if an issue occurs when either generating a TOON
    structure or normalized Pydantic model representation of a user
    provided output type.
    """

    pass


class ToonAIRequestError(ToonAIError):
    """
    Exception raised if an issue occurs during the invocation of an
    LLM using LiteLLM.
    """

    pass


class ToonAIResponseError(ToonAIError):
    """
    Exception raised if an issue occurs during the decoding of a standard
    or streamed response from an LLM to a user specified output type.
    """

    def __init__(
        self,
        message: str,
        raw_response: Any | None = None,
        raw_text: str | None = None,
    ):
        if raw_response:
            message = f"{message}\n\nRaw response: {raw_response}"
        if raw_text:
            message = f"{message}\n\nRaw text: {raw_text}"

        super().__init__(message)
        self.raw_response = raw_response
        self.raw_text = raw_text
