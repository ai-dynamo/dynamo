#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import uuid

import base58
from sqlmodel import Field as SQLField
from sqlmodel import UniqueConstraint

from components import CompoundNimBase, CompoundNimVersionBase

"""
This file stores all of the models/tables stored in the SQL database.
This is needed because otherwise we get an error like so:

raise exc.InvalidRequestError(sqlalchemy.exc.InvalidRequestError:
When initializing mapper Mapper[Checkpoint(checkpoint)],
expression "relationship("Optional['Model']")" seems to be using a generic class as the
argument to relationship(); please state the generic argument using an annotation, e.g.
"parent_model: Mapped[Optional['Model']] = relationship()"
"""


def get_random_id(prefix: str) -> str:
    u = uuid.uuid4()
    return f"{prefix}-{base58.b58encode(u.bytes).decode('ascii')}"


def new_compound_entity_id() -> str:
    return get_random_id("compound")


class CompoundNimVersion(CompoundNimVersionBase, table=True):
    """A row in the compoundai nim table."""

    __table_args__ = (
        UniqueConstraint("compound_nim_id", "version", name="version_unique_per_nim"),
    )

    id: str = SQLField(default_factory=new_compound_entity_id, primary_key=True)

    compound_nim_id: str = SQLField(foreign_key="compoundnim.id")


class CompoundNim(CompoundNimBase, table=True):
    """A row in the compoundai nim table."""

    id: str = SQLField(default_factory=new_compound_entity_id, primary_key=True)
