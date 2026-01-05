"""Shared types for GPU Memory Service.

Enums and dataclasses used across multiple modules.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import FrozenSet


class ServerState(str, Enum):
    """Server state - derived from actual connections.

    State Diagram:

        EMPTY ──RW_CONNECT──► RW ──RW_COMMIT──► COMMITTED
          ▲                    │                   │
          │                    │                   │
          └───RW_ABORT─────────┘                   │
                                                   ▼
        COMMITTED ◄──RO_DISCONNECT (last)── RO ◄──RO_CONNECT
                          │                  ▲
                          │                  │
                          └──RO_CONNECT──────┘
                          └──RO_DISCONNECT───┘ (not last)
    """

    EMPTY = "EMPTY"  # No connections, not committed
    RW = "RW"  # Writer connected (exclusive)
    COMMITTED = "COMMITTED"  # No connections, committed (weights valid)
    RO = "RO"  # Reader(s) connected (shared)


class ConnectionMode(str, Enum):
    """Connection lock mode."""

    RW = "rw"  # Writer (exclusive)
    RO = "ro"  # Reader (shared)


class StateEvent(Enum):
    """Events that trigger state transitions."""

    RW_CONNECT = auto()  # Writer connects
    RW_COMMIT = auto()  # Writer calls commit()
    RW_ABORT = auto()  # Writer disconnects without commit
    RO_CONNECT = auto()  # Reader connects
    RO_DISCONNECT = auto()  # Reader disconnects


class Operation(Enum):
    """Operations that can be performed on the server.

    Each operation has permission requirements:
    - Required connection mode (RW or RO or either)
    - Allowed server states
    """

    # Allocation operations
    ALLOCATE = auto()  # Create physical memory allocation
    EXPORT = auto()  # Export allocation as POSIX FD
    FREE = auto()  # Free single allocation
    CLEAR_ALL = auto()  # Clear all allocations
    GET_ALLOCATION = auto()  # Get allocation info
    LIST_ALLOCATIONS = auto()  # List all allocations

    # Metadata store operations
    METADATA_PUT = auto()  # Store metadata entry
    METADATA_GET = auto()  # Get metadata entry
    METADATA_DELETE = auto()  # Delete metadata entry
    METADATA_LIST = auto()  # List metadata keys

    # State queries (always allowed)
    GET_LOCK_STATE = auto()  # Get lock/session state
    GET_ALLOCATION_STATE = auto()  # Get allocation state

    # Lifecycle
    COMMIT = auto()  # Publish and release RW lock


# Permission definitions: which operations require RW mode
RW_REQUIRED_OPS: FrozenSet[Operation] = frozenset(
    {
        Operation.ALLOCATE,
        Operation.FREE,
        Operation.CLEAR_ALL,
        Operation.METADATA_PUT,
        Operation.METADATA_DELETE,
        Operation.COMMIT,
    }
)

# Operations allowed in each state (with appropriate connection mode)
# RW state: only RW connection can be present
# RO state: only RO connections can be present
STATE_ALLOWED_OPS: dict[ServerState, FrozenSet[Operation]] = {
    ServerState.EMPTY: frozenset(),  # No connections possible in EMPTY
    ServerState.RW: frozenset(
        {
            Operation.ALLOCATE,
            Operation.EXPORT,
            Operation.FREE,
            Operation.CLEAR_ALL,
            Operation.GET_ALLOCATION,
            Operation.LIST_ALLOCATIONS,
            Operation.METADATA_PUT,
            Operation.METADATA_GET,
            Operation.METADATA_DELETE,
            Operation.METADATA_LIST,
            Operation.GET_LOCK_STATE,
            Operation.GET_ALLOCATION_STATE,
            Operation.COMMIT,
        }
    ),
    ServerState.COMMITTED: frozenset(),  # No connections possible in COMMITTED
    ServerState.RO: frozenset(
        {
            Operation.EXPORT,
            Operation.GET_ALLOCATION,
            Operation.LIST_ALLOCATIONS,
            Operation.METADATA_GET,
            Operation.METADATA_LIST,
            Operation.GET_LOCK_STATE,
            Operation.GET_ALLOCATION_STATE,
        }
    ),
}


@dataclass
class StateSnapshot:
    """Current server state snapshot."""

    state: ServerState
    has_rw: bool
    ro_count: int
    waiting_writers: int
    committed: bool

    @property
    def is_ready(self) -> bool:
        """Ready = committed and no RW connection."""
        return self.committed and not self.has_rw


def derive_state(has_rw: bool, ro_count: int, committed: bool) -> ServerState:
    """Derive server state from connection info."""
    if has_rw:
        return ServerState.RW
    if ro_count > 0:
        return ServerState.RO
    if committed:
        return ServerState.COMMITTED
    return ServerState.EMPTY
