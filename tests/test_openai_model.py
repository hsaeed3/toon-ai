"""tests.test_openai_model"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from toon_ai import ToonClient


class User(BaseModel):
    name: str
    age: int
    is_student: bool


def test_openai_model_structured_type() -> None:
    client = ToonClient()

    response = client.generate(
        messages=[
            {"role": "user", "content": "John is 20 and he is a student"}
        ],
        output_type=User,
        model="openai/gpt-4o-mini",
    )
    assert (
        response.model_dump()
        == User(
            name="John",
            age=20,
            is_student=True,
        ).model_dump()
    )


def test_openai_reasoning_model_structured_type() -> None:
    client = ToonClient()

    response = client.generate(
        messages=[
            {"role": "user", "content": "John is 20 and he is a student"}
        ],
        output_type=User,
        model="openai/gpt-5-mini",
    )
    assert (
        response.model_dump()
        == User(
            name="John",
            age=20,
            is_student=True,
        ).model_dump()
    )


def test_openai_model_primitive_type() -> None:
    client = ToonClient()

    response = client.generate(
        messages=[{"role": "user", "content": "What is 20+20?"}],
        output_type=int,
        model="openai/gpt-4o-mini",
    )
    assert response == 40


def test_openai_reasoning_model_primitive_type() -> None:
    client = ToonClient()

    response = client.generate(
        messages=[{"role": "user", "content": "What is 20+20?"}],
        output_type=int,
        model="openai/gpt-5-mini",
    )
    assert response == 40


if __name__ == "__main__":
    pytest.main([__file__])
