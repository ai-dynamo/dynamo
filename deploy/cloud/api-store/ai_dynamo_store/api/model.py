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
from datetime import datetime, timezone
from typing import Optional

import base58
from sqlalchemy import Column, DateTime
from sqlmodel import Field as SQLField
from sqlmodel import UniqueConstraint

from .components import DynamoComponentBase, DynamoComponentVersionBase

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


# Define a function to create timezone-naive datetime objects
def utc_now_naive() -> datetime:
    """Return current UTC time without timezone info for database compatibility"""
    now = datetime.now(timezone.utc)
    return now.replace(tzinfo=None)


# Utility function to strip timezone info from datetime objects
def make_naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert a datetime to naive (no timezone) if it has timezone info"""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


# Utility function to add UTC timezone to naive datetime objects
def make_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Add UTC timezone to naive datetime objects"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class DynamoComponentVersion(DynamoComponentVersionBase, table=True):
    """A row in the dynamo nim table."""

    __tablename__ = "dynamonimversion"
    __table_args__ = (
        UniqueConstraint(
            "dynamo_component_id", "version", name="version_unique_per_nim"
        ),
    )

    id: str = SQLField(default_factory=new_compound_entity_id, primary_key=True)

    # Override the datetime fields to explicitly use timezone-naive datetimes
    # created_at: datetime = SQLField(
    #     sa_column=Column(DateTime, nullable=False, default=utc_now_naive)
    # )
    # updated_at: datetime = SQLField(
    #     sa_column=Column(
    #         DateTime, nullable=False, default=utc_now_naive, onupdate=utc_now_naive
    #     )
    # )
    # upload_started_at: datetime = SQLField(sa_column=Column(DateTime, nullable=True))
    # upload_finished_at: datetime = SQLField(sa_column=Column(DateTime, nullable=True))
    build_at: datetime = SQLField(sa_column=Column(DateTime, nullable=False))

    dynamo_component_id: str = SQLField(foreign_key="dynamonim.id")


class DynamoComponent(DynamoComponentBase, table=True):
    """A row in the dynamo nim table."""

    __tablename__ = "dynamonim"

    id: str = SQLField(default_factory=new_compound_entity_id, primary_key=True)

    # Override the datetime fields to explicitly use timezone-naive datetimes
    # created_at: datetime = SQLField(
    #     sa_column=Column(DateTime, nullable=False, default=utc_now_naive)
    # )
    # updated_at: datetime = SQLField(
    #     sa_column=Column(
    #         DateTime, nullable=False, default=utc_now_naive, onupdate=utc_now_naive
    #     )
    # )
    # deleted_at: datetime = SQLField(sa_column=Column(DateTime, nullable=True))
