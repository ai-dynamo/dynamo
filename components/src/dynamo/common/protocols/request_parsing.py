# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a media request model from a request body.

The frontend checks a request before it forwards it to a worker, and in a
rolling upgrade an older frontend forwards what a newer model rejects. The
worker then answers with a 400 that names each rejected field and the rule
it broke, and not the value: ``str(ValidationError)`` carries
``input_value``, which can be the prompt or inline image data.

This module imports the Dynamo runtime for ``InvalidArgument``, so it stays
out of the package ``__init__``: the generated models import without it.
"""

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from dynamo.llm.exceptions import InvalidArgument

RequestT = TypeVar("RequestT", bound=BaseModel)


def parse_media_request(model: type[RequestT], request: Mapping[str, Any]) -> RequestT:
    """Return ``request`` parsed as ``model``, or raise ``InvalidArgument``.

    The error message is one ``field: rule`` entry per rejected field, as
    :func:`rejected_fields` writes it. The binding returns it to the client
    as a 400.
    """
    try:
        return model.model_validate(request)
    except ValidationError as error:
        raise InvalidArgument(rejected_fields(error)) from error


def rejected_fields(error: ValidationError) -> str:
    """The fields ``error`` rejected, each with the rule it broke.

    ``nvext.fps: Input should be greater than or equal to 1``, joined by
    ``; `` when there are several. The rule text is pydantic's, built from
    the constraint (a bound, the ``Literal`` values, the expected type) and
    not from the input. An error on the body itself, such as a body that is
    not an object, is named ``request``.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'request'}: {item['msg']}"
        for item in error.errors()
    )
