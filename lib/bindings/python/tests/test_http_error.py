import pytest

from dynamo.llm import HttpError


def test_raise_http_error():
    with pytest.raises(HttpError):
        raise HttpError(404, "Not Found")
    with pytest.raises(Exception):
        raise HttpError(500, "Internal Server Error")


def test_invalid_http_error_code():
    with pytest.raises(AssertionError):
        HttpError(1700, "Invalid Code")
