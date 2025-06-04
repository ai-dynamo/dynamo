# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from dynamo.sdk import (
    AbstractDynamoService,
    abstract_dynamo_endpoint,
    endpoint,
    service,
)


class TestInterface(AbstractDynamoService):
    @abstract_dynamo_endpoint
    def foo(self, x: int) -> int:
        pass

    @abstract_dynamo_endpoint
    def bar(self, y: int) -> int:
        pass


class ExtraInterface(AbstractDynamoService):
    @abstract_dynamo_endpoint
    def baz(self, z: int) -> int:
        pass


@service
class GoodService(TestInterface, ExtraInterface):
    @endpoint
    def foo(self, x: int) -> int:
        return x + 1

    @endpoint
    def bar(self, y: int) -> int:
        return y + 1

    @endpoint
    def baz(self, z: int) -> int:
        return z + 1


@service
class ParentService(TestInterface):
    @endpoint
    def foo(self, x: int) -> int:
        return x + 1

    @endpoint
    def bar(self, y: int) -> int:
        return y + 1


@service
class GrandChildService(ParentService):
    pass


def test_good_service():
    """Test that a properly implemented service works."""
    svc = GoodService()
    assert svc.foo(1) == 2
    assert svc.bar(2) == 3
    assert svc.baz(3) == 4


def test_inheritance():
    """Test that inheritance works correctly."""
    svc = GrandChildService()
    assert svc.foo(1) == 2
    assert svc.bar(2) == 3


def test_missing_implementation():
    """Test that missing implementations are caught."""
    with pytest.raises(TypeError) as exc_info:

        @service
        class MissingEndpointService(TestInterface, ExtraInterface):
            @endpoint
            def foo(self, x: int) -> int:
                return x + 1

            @endpoint
            def bar(self, y: int) -> int:
                return y + 1

    assert "missing implementation(s): baz" in str(exc_info.value)


def test_undecorated_method():
    """Test that undecorated methods are caught."""
    with pytest.raises(TypeError) as exc_info:

        @service
        class UndecoratedService(TestInterface):
            def foo(self, x: int) -> int:
                return x + 1

            @endpoint
            def bar(self, y: int) -> int:
                return y + 1

    assert "method(s) not decorated with @endpoint: foo" in str(exc_info.value)


def test_non_callable():
    """Test that non-callable attributes are caught."""
    with pytest.raises(TypeError) as exc_info:

        @service
        class ShadowedService(ParentService):
            bar = 123  # hides the callable

    assert "bar must be callable, got int" in str(exc_info.value)


def test_interface_instantiation():
    """Test that interfaces cannot be instantiated."""
    with pytest.raises(TypeError):
        TestInterface()
