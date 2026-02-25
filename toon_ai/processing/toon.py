"""toon_ai.processing.toon"""

from __future__ import annotations

import logging
import typing
from typing import (
    Any,
    Type,
    TypeVar,
)

from pydantic import BaseModel, ValidationError
from toon_format import (
    decode,
)

from ..core.logger import _log_debug_context
from ..core.exceptions import ToonAIResponseError, ToonAITypeError
from ..core.requests import ToonClientRequestParams


OutputType = TypeVar("OutputType")


_logger = logging.getLogger("toon_ai.processing.toon")


def _format_type_for_toon(annotation: Any, description: str) -> str:
    """Format a type annotation for TOON structure display."""
    from enum import Enum

    origin = getattr(annotation, "__origin__", None)

    if origin is typing.Annotated:
        args = getattr(annotation, "__args__", ())
        if args:
            return _format_type_for_toon(args[0], description)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        choices = "|".join(str(m.value) for m in annotation)
        return f"<{choices}>"

    if origin is typing.Literal:
        choices = "|".join(str(v) for v in getattr(annotation, "__args__", ()))
        return f"<{choices}>"

    if origin is typing.Union:
        args = [
            arg
            for arg in getattr(annotation, "__args__", ())
            if arg is not type(None)
        ]
        type_names = []
        for t in args:
            if t is str:
                type_names.append("str")
            elif t is int:
                type_names.append("int")
            elif t is float:
                type_names.append("float")
            elif t is bool:
                type_names.append("bool")
            elif isinstance(t, type) and issubclass(t, Enum):
                type_names.append("|".join(str(m.value) for m in t))
            else:
                type_names.append(
                    str(t.__name__) if hasattr(t, "__name__") else str(t)
                )
        return f"<{' or '.join(type_names)}>"

    if annotation is str:
        return f'"<str>"'
    elif annotation is int:
        return "<int>"
    elif annotation is float:
        return "<float>"
    elif annotation is bool:
        return "<bool>"
    else:
        return f"<{description}>"


def _coerce_enums_for_model(
    model: type[BaseModel], data: dict[str, Any]
) -> dict[str, Any]:
    """Coerce string values to enum instances for fields that expect enums."""
    from enum import Enum
    import typing

    if not isinstance(data, dict):
        return data

    result = data.copy()

    for field_name, field_info in model.model_fields.items():
        if field_name not in result:
            continue

        annotation = field_info.annotation
        origin = getattr(annotation, "__origin__", None)

        if origin is typing.Union or str(origin) == "typing.Union":
            args = getattr(annotation, "__args__", ())
            for arg in args:
                if (
                    arg is not type(None)
                    and isinstance(arg, type)
                    and issubclass(arg, Enum)
                ):
                    annotation = arg
                    break

        if isinstance(annotation, type) and issubclass(annotation, Enum):
            value = result[field_name]
            if isinstance(value, str):
                for member in annotation:
                    if member.value == value or member.name == value:
                        result[field_name] = member
                        break

        elif isinstance(annotation, type) and issubclass(
            annotation, BaseModel
        ):
            if isinstance(result[field_name], dict):
                result[field_name] = _coerce_enums_for_model(
                    annotation, result[field_name]
                )

        elif origin is list:
            args = getattr(annotation, "__args__", ())
            if (
                args
                and isinstance(args[0], type)
                and issubclass(args[0], BaseModel)
            ):
                item_model = args[0]
                if isinstance(result[field_name], list):
                    result[field_name] = [
                        _coerce_enums_for_model(item_model, item)
                        if isinstance(item, dict)
                        else item
                        for item in result[field_name]
                    ]

    return result


def _coerce_output_value(
    request_params: ToonClientRequestParams[OutputType],
    partial_model: BaseModel,
) -> OutputType:
    if isinstance(request_params.output_type, type) and not issubclass(
        request_params.output_type, BaseModel
    ):
        if hasattr(partial_model, "content"):
            return getattr(partial_model, "content")
    return partial_model  # type: ignore[return-value]


def encode_toon_structure(model: type[Any], indent: int = 0) -> str:
    """
    Generate a TOON structure template from a Pydantic model.

    Recursively expands nested Pydantic models to show full structure.
    Handles Enums, Literals, Unions, Annotated, and nested types.

    Args:
        model: A Pydantic BaseModel class
        indent: Current indentation level

    Returns:
        A string representing the TOON structure template
    """
    from pydantic import BaseModel

    prefix = "  " * indent
    lines = []

    try:
        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation
            description = field_info.description or f"value for {field_name}"

            origin = getattr(annotation, "__origin__", None)
            if origin is typing.Annotated:
                args = getattr(annotation, "__args__", ())
                if args:
                    annotation = args[0]
                    origin = getattr(annotation, "__origin__", None)

            original_annotation = annotation

            if origin is type(None) or str(origin) == "typing.Union":
                args = getattr(annotation, "__args__", ())
                non_none_args = [arg for arg in args if arg is not type(None)]
                if len(non_none_args) == 1:
                    annotation = non_none_args[0]
                    original_annotation = non_none_args[0]
                elif len(non_none_args) > 1:
                    formatted = _format_type_for_toon(annotation, description)
                    lines.append(f"{prefix}{field_name}: {formatted}")
                    continue

            if isinstance(annotation, type) and issubclass(
                annotation, BaseModel
            ):
                lines.append(f"{prefix}{field_name}:")
                nested_structure = encode_toon_structure(
                    annotation, indent + 1
                )
                lines.append(nested_structure)
            elif origin is list:
                args = getattr(annotation, "__args__", ())
                if (
                    args
                    and isinstance(args[0], type)
                    and issubclass(args[0], BaseModel)
                ):
                    item_model = args[0]
                    has_nested_list = any(
                        getattr(f.annotation, "__origin__", None) is list
                        for f in item_model.model_fields.values()
                    )
                    if has_nested_list:
                        lines.append(f"{prefix}{field_name}[N]:")
                        lines.append(f"{prefix}  - <item>:")
                        nested = encode_toon_structure(item_model, indent + 2)
                        lines.append(nested)
                        lines.append(f"{prefix}  ...")
                    else:
                        item_fields = list(item_model.model_fields.keys())
                        headers = ",".join(item_fields)
                        lines.append(f"{prefix}{field_name}[N,]{{{headers}}}:")
                        placeholders = ",".join(
                            f"<{item_model.model_fields[f].description or f}>"
                            for f in item_fields
                        )
                        lines.append(f"{prefix}  {placeholders}")
                        lines.append(f"{prefix}  ...")
                else:
                    lines.append(
                        f"{prefix}{field_name}[N]: <value>,<value>,..."
                    )
            elif origin is dict or annotation is dict:
                lines.append(f"{prefix}{field_name}:")
                lines.append(f"{prefix}  <key>: <value>")
            else:
                formatted = _format_type_for_toon(
                    original_annotation, description
                )
                lines.append(f"{prefix}{field_name}: {formatted}")

    except Exception as e:
        raise ToonAITypeError(
            f"Failed to encode normalized Pydantic model representation of output type into TOON structure: {e}",
        ) from e

    content = "\n".join(lines)

    _log_debug_context(
        logger=_logger,
        lines=[
            f"Encoded normalized Pydantic model representation of output type into TOON structure: [italic]{content}[/italic].",
        ],
    )
    return content


def decode_toon(
    normalized_type: Type[BaseModel],
    toon_content: str,
) -> BaseModel | None:
    """
    Decodes text containing a TOON formatted content into the 'normalized' or Pydantic model
    representation of a user provided output type.

    Args:
        normalized_type : Type[BaseModel]
            The normalized Pydantic model representation of the user provided output type.
        toon_content : str
            A string containing a code block representation of a TOON structure.

    Returns:
        A Pydantic model instance
    """
    if not toon_content:
        raise ToonAIResponseError(
            "No TOON content found in response",
            raw_text=toon_content if toon_content else "",
        )

    try:
        data = decode(toon_content)
        data = (
            _coerce_enums_for_model(normalized_type, data)
            if isinstance(data, dict)
            else data
        )
    except Exception as e:
        raise ToonAIResponseError(
            f"Failed to parse TOON content: {e}",
            raw_text=toon_content if toon_content else "",
        ) from e

    try:
        return normalized_type.model_validate(data)
    except ValidationError as e:
        raise ToonAIResponseError(
            f"Failed to validate TOON content: {e}",
            raw_text=toon_content if toon_content else "",
        ) from e
