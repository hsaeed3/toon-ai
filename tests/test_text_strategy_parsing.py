"""tests.test_response_parsing"""

from __future__ import annotations

import pytest

from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, TYPE_CHECKING

from pydantic import BaseModel

from toon_ai.core.requests import ToonClientRequestParams
from toon_ai.strategies.text import TextStrategy

if TYPE_CHECKING:
    from litellm import ModelResponse


class User(BaseModel):
    name: str
    age: int
    is_student: bool


@dataclass
class MockModelResponse:
    @dataclass
    class Choice:
        @dataclass
        class Message:
            content: str

        message: Message

    choices: List[Choice]


def mock_request_params() -> ToonClientRequestParams[User]:
    return ToonClientRequestParams(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is the name of the user?"}
        ],
        output_type=User,
    )


def mock_model_response_correct() -> ModelResponse:
    return MockModelResponse(
        choices=[
            MockModelResponse.Choice(
                message=MockModelResponse.Choice.Message(
                    content="```toon\nname: John\nage: 20\nis_student: true\n```"
                )
            )
        ]
    )  # type: ignore[return-value]


def mock_model_response_correct_nested() -> ModelResponse:
    return MockModelResponse(
        choices=[
            MockModelResponse.Choice(
                message=MockModelResponse.Choice.Message(
                    content="Here is your expected response: ```toon\nname: John\nage: 20\nis_student: true\n``` . Did you find it?"
                )
            )
        ]
    )  # type: ignore[return-value]


def test_parse_toon_block_from_model_response_correct() -> None:
    validated = TextStrategy().parse_response(
        request_params=mock_request_params(),
        response=mock_model_response_correct(),
    )
    assert (
        validated.model_dump()
        == User(
            name="John",
            age=20,
            is_student=True,
        ).model_dump()
    )


def test_parse_toon_block_from_model_response_correct_nested() -> None:
    validated = TextStrategy().parse_response(
        request_params=mock_request_params(),
        response=mock_model_response_correct_nested(),
    )
    assert (
        validated.model_dump()
        == User(
            name="John",
            age=20,
            is_student=True,
        ).model_dump()
    )


if __name__ == "__main__":
    pytest.main([__file__])
