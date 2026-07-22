# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client helpers for GMS KV block leases."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import mmap
import os
import struct
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_KV_LEASE_SHM_MAGIC = 0x4C534D47
_KV_LEASE_SHM_VERSION = 1
_KV_LEASE_SHM_HEADER_SIZE = 64
# Per-block record: state (u32) | generation (u32) | owner_hash (u64).
_KV_LEASE_SHM_RECORD_SIZE = 16
_KV_LEASE_SHM_HEADER_STRUCT = struct.Struct("<IIIIQQ")

logger = logging.getLogger(__name__)
_LEASE_PRESSURE_LOG_STATE: dict[str, tuple[float, int]] = {}
_LEASE_PRESSURE_LOG_LOCK = threading.Lock()


def _lease_pressure_log_interval_s() -> float:
    raw = os.environ.get("GMS_KV_LEASE_PRESSURE_LOG_INTERVAL_S", "10")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 10.0


def log_lease_pressure(
    target_logger: logging.Logger,
    key: str,
    message: str,
    **fields: object,
) -> None:
    """Rate-limited warning for lease pressure and anomalous lease events.

    Allocation failures can happen on every scheduler iteration under high KV
    pressure. Production triage still needs the first event and periodic state,
    so this helper logs immediately and then coalesces repeated messages by key.
    Set GMS_KV_LEASE_PRESSURE_LOG_INTERVAL_S=0 to log every event.
    """
    interval = _lease_pressure_log_interval_s()
    now = time.monotonic()
    suppressed = 0
    with _LEASE_PRESSURE_LOG_LOCK:
        last, suppressed = _LEASE_PRESSURE_LOG_STATE.get(key, (0.0, 0))
        should_log = interval == 0.0 or last == 0.0 or now - last >= interval
        if should_log:
            _LEASE_PRESSURE_LOG_STATE[key] = (now, 0)
        else:
            _LEASE_PRESSURE_LOG_STATE[key] = (last, suppressed + 1)
            return
    if suppressed:
        fields = {**fields, "suppressed": suppressed}
    rendered = " ".join(f"{name}={value!r}" for name, value in fields.items())
    if rendered:
        target_logger.warning("%s %s", message, rendered)
    else:
        target_logger.warning("%s", message)


def _env_enabled(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def resolve_lease_device(
    env_name: str,
    *,
    fallback_env_names: tuple[str, ...] = ("LOCAL_RANK",),
) -> int:
    for name in (env_name, *fallback_env_names):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r for GMS KV leases", name, value)
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        logger.debug("Unable to derive CUDA device for GMS KV leases", exc_info=True)
    return 0


def _kv_lease_shm_path(engine: str, namespace: str) -> str:
    engine_upper = engine.upper().replace("-", "_")
    explicit = os.environ.get(f"GMS_{engine_upper}_KV_LEASE_SHM_PATH")
    if explicit is None:
        explicit = os.environ.get("GMS_KV_LEASE_SHM_PATH")
    if explicit:
        return explicit

    base_dir = os.environ.get(f"GMS_{engine_upper}_KV_LEASE_SHM_DIR")
    if base_dir is None:
        base_dir = os.environ.get("GMS_KV_LEASE_SHM_DIR")
    if not base_dir:
        base_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:20]
    return os.path.join(base_dir, f"gms-kv-lease-{digest}.shm")


def _owner_hash(owner_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(owner_id.encode("utf-8")).digest()[:8], "little"
    )


def _read_shm_header(fd: int) -> tuple[int, int, int, int, int, int] | None:
    try:
        data = os.pread(fd, _KV_LEASE_SHM_HEADER_SIZE, 0)
    except OSError:
        return None
    if len(data) < _KV_LEASE_SHM_HEADER_STRUCT.size:
        return None
    return _KV_LEASE_SHM_HEADER_STRUCT.unpack_from(data)


def _valid_shm_header(header: tuple[int, int, int, int, int, int] | None) -> bool:
    if header is None:
        return False
    magic, version, total_blocks, record_size, _free_count, _reserved = header
    return (
        magic == _KV_LEASE_SHM_MAGIC
        and version == _KV_LEASE_SHM_VERSION
        and total_blocks > 0
        and record_size == _KV_LEASE_SHM_RECORD_SIZE
    )


@dataclass(frozen=True)
class KVLease:
    block_id: int
    generation: int


class KVLeaseClient(Protocol):
    namespace: str
    owner_id: str

    def acquire(
        self,
        count: int,
        *,
        preferred_blocks: list[int] | None = None,
        allow_partial: bool = False,
        strict_preferred: bool = False,
    ) -> list[KVLease]:
        ...

    def seal(self, leases: list[KVLease]) -> None:
        ...

    def adopt(self, leases: list[KVLease]) -> list[KVLease]:
        ...

    def release(self, leases: list[KVLease]) -> None:
        ...

    def free_count(self) -> int:
        ...


class SharedMemoryKVLeaseClient:
    """KV lease client backed by a shared mmap and native atomics.

    This is the only engine-integration lease path. Block acquire/release and
    free-count reads are local shared-memory operations, so scheduler decisions
    do not scale with RPC round trips or namespace size.
    """

    def __init__(
        self,
        shm_path: str,
        *,
        namespace: str,
        owner_id: str,
        total_blocks: int,
        reserved_blocks: list[int] | None = None,
    ) -> None:
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive")
        self.namespace = namespace
        self.owner_id = owner_id
        self.total_blocks = int(total_blocks)
        self.shm_path = shm_path
        self._owner_hash = _owner_hash(owner_id)
        self._rust = self._load_rust_ring()
        self._fd, self._mmap = self._open_or_init(
            shm_path, int(total_blocks), reserved_blocks or []
        )

    @classmethod
    def from_env(
        cls,
        engine: str,
        device: int,
        *,
        total_blocks: int,
        namespace: str | None = None,
        owner_id: str | None = None,
        namespace_suffix: str = "kv",
        reserved_blocks: list[int] | None = None,
    ) -> "SharedMemoryKVLeaseClient":
        engine_upper = engine.upper().replace("-", "_")
        if namespace is None:
            namespace = os.environ.get(
                f"GMS_{engine_upper}_KV_LEASE_NAMESPACE",
                os.environ.get(
                    "GMS_KV_LEASE_NAMESPACE",
                    f"{engine}:gpu{device}:{namespace_suffix}",
                ),
            )
        if owner_id is None:
            owner_id = os.environ.get(
                f"GMS_{engine_upper}_KV_LEASE_OWNER_ID",
                os.environ.get(
                    "GMS_KV_LEASE_OWNER_ID", f"{engine}-{os.getpid()}-{device}"
                ),
            )
        return cls(
            _kv_lease_shm_path(engine, namespace),
            namespace=namespace,
            owner_id=owner_id,
            total_blocks=total_blocks,
            reserved_blocks=reserved_blocks,
        )

    @staticmethod
    def _load_rust_ring():
        import gms_rust_ring  # type: ignore[import-not-found]

        required = (
            "kv_lease_init",
            "kv_lease_free_count",
            "kv_lease_acquire",
            "kv_lease_seal",
            "kv_lease_adopt",
            "kv_lease_release",
            "kv_lease_reclaim_foreign",
        )
        missing = [name for name in required if not hasattr(gms_rust_ring, name)]
        if missing:
            raise RuntimeError(
                f"gms_rust_ring missing KV lease functions: {', '.join(missing)}"
            )
        # The Rust ring owns the shm layout; the Python constants above only
        # mirror it for the no-ring parse paths. Fail loudly at load if they ever
        # drift, rather than silently misparsing every header/record.
        for py_value, rust_attr in (
            (_KV_LEASE_SHM_MAGIC, "KV_LEASE_MAGIC"),
            (_KV_LEASE_SHM_VERSION, "KV_LEASE_VERSION"),
            (_KV_LEASE_SHM_HEADER_SIZE, "KV_LEASE_HEADER_SIZE"),
            (_KV_LEASE_SHM_RECORD_SIZE, "KV_LEASE_RECORD_SIZE"),
        ):
            rust_value = getattr(gms_rust_ring, rust_attr, None)
            if rust_value is not None and int(rust_value) != int(py_value):
                raise RuntimeError(
                    "gms_rust_ring KV lease layout drift: "
                    f"{rust_attr}={rust_value} but Python expects {py_value}"
                )
        return gms_rust_ring

    @staticmethod
    def _reset_requested() -> bool:
        return _env_enabled("GMS_KV_LEASE_SHM_RESET", False)

    def _open_or_init(
        self,
        shm_path: str,
        total_blocks: int,
        reserved_blocks: list[int],
    ) -> tuple[int, mmap.mmap]:
        os.makedirs(os.path.dirname(os.path.abspath(shm_path)), exist_ok=True)
        fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o600)
        locked = False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
            header = _read_shm_header(fd)
            stat = os.fstat(fd)
            should_init = self._reset_requested() or not _valid_shm_header(header)
            if should_init:
                map_blocks = int(total_blocks)
                map_size = (
                    _KV_LEASE_SHM_HEADER_SIZE + map_blocks * _KV_LEASE_SHM_RECORD_SIZE
                )
                os.ftruncate(fd, map_size)
                buf = mmap.mmap(fd, map_size)
                self._rust.kv_lease_init(buf, map_blocks, reserved_blocks)
                return fd, buf

            assert header is not None
            _magic, _version, existing_blocks, record_size, _free_count, _epoch = header
            if record_size != _KV_LEASE_SHM_RECORD_SIZE:
                raise RuntimeError(
                    f"unsupported KV lease record size in {shm_path}: {record_size}"
                )
            if int(existing_blocks) != int(total_blocks):
                raise RuntimeError(
                    "KV lease shared-memory size mismatch: "
                    f"path={shm_path} existing={existing_blocks} requested={total_blocks}"
                )
            map_size = (
                _KV_LEASE_SHM_HEADER_SIZE
                + int(existing_blocks) * _KV_LEASE_SHM_RECORD_SIZE
            )
            if stat.st_size < map_size:
                os.ftruncate(fd, map_size)
            buf = mmap.mmap(fd, map_size)
            return fd, buf
        except Exception:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
                locked = False
            os.close(fd)
            raise
        finally:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)

    def close(self) -> None:
        try:
            self._mmap.close()
        finally:
            os.close(self._fd)

    def acquire(
        self,
        count: int,
        *,
        preferred_blocks: list[int] | None = None,
        allow_partial: bool = False,
        strict_preferred: bool = False,
    ) -> list[KVLease]:
        requested = int(count)
        if requested <= 0:
            return []

        try:
            infos = self._rust.kv_lease_acquire(
                self._mmap,
                [int(block_id) for block_id in preferred_blocks or []],
                requested,
                bool(allow_partial),
                bool(strict_preferred),
                int(self._owner_hash),
            )
        except Exception as exc:
            self._log_acquire_failure(
                requested=requested,
                preferred_blocks=preferred_blocks,
                allow_partial=allow_partial,
                strict_preferred=strict_preferred,
                available=self.free_count(),
                error=type(exc).__name__,
            )
            raise
        if len(infos) != requested and not allow_partial:
            self._log_acquire_short(
                requested=requested,
                returned=len(infos),
                preferred_blocks=preferred_blocks,
                strict_preferred=bool(strict_preferred),
            )
        return self._lease_infos_to_records(infos)

    @staticmethod
    def _lease_infos_to_records(infos) -> list[KVLease]:
        return [
            KVLease(int(block_id), int(generation)) for block_id, generation in infos
        ]

    def _log_acquire_failure(
        self,
        *,
        requested: int,
        preferred_blocks: list[int] | None,
        allow_partial: bool,
        strict_preferred: bool,
        available: int,
        error: str,
    ) -> None:
        log_lease_pressure(
            logger,
            f"{self.namespace}:acquire-error",
            "GMS KV lease acquire failed",
            namespace=self.namespace,
            owner_id=self.owner_id,
            requested=requested,
            available=available,
            raw_free=self.raw_free_count(),
            total_blocks=self.total_blocks,
            preferred_count=len(preferred_blocks or []),
            allow_partial=bool(allow_partial),
            strict_preferred=bool(strict_preferred),
            error=error,
        )

    def _log_acquire_short(
        self,
        *,
        requested: int,
        returned: int,
        preferred_blocks: list[int] | None,
        strict_preferred: bool,
    ) -> None:
        log_lease_pressure(
            logger,
            f"{self.namespace}:acquire-short",
            "GMS KV lease acquire returned fewer leases than requested",
            namespace=self.namespace,
            owner_id=self.owner_id,
            requested=requested,
            returned=returned,
            free_count=self.free_count(),
            total_blocks=self.total_blocks,
            preferred_count=len(preferred_blocks or []),
            strict_preferred=bool(strict_preferred),
        )

    def seal(self, leases: list[KVLease]) -> None:
        if not leases:
            return
        self._rust.kv_lease_seal(
            self._mmap,
            [int(lease.block_id) for lease in leases],
            [int(lease.generation) for lease in leases],
        )

    def adopt(self, leases: list[KVLease]) -> list[KVLease]:
        if not leases:
            return []
        result = self._rust.kv_lease_adopt(
            self._mmap,
            [int(lease.block_id) for lease in leases],
            [int(lease.generation) for lease in leases],
            int(self._owner_hash),
        )
        if result is None:
            return []
        return [
            KVLease(int(block_id), int(generation)) for block_id, generation in result
        ]

    def release(self, leases: list[KVLease]) -> None:
        if not leases:
            return
        try:
            self._rust.kv_lease_release(
                self._mmap,
                [int(lease.block_id) for lease in leases],
                [int(lease.generation) for lease in leases],
            )
        except Exception as exc:
            log_lease_pressure(
                logger,
                f"{self.namespace}:release-error",
                "GMS KV lease release failed",
                namespace=self.namespace,
                owner_id=self.owner_id,
                count=len(leases),
                first_block=int(leases[0].block_id),
                error=type(exc).__name__,
            )
            raise

    def reclaim_foreign(
        self,
        *,
        max_blocks: int = 0,
        protected_blocks: set[int] | None = None,
    ) -> int:
        """Recover and release orphaned records after prior writers are fenced.

        The caller must ensure no prior writer can resume and must invoke this
        before starting mutations for the replacement owner. Recovery resolves
        interrupted acquire/release transitions and reconstructs the free count
        from record state before returning.
        """
        protected = sorted(int(block_id) for block_id in (protected_blocks or set()))
        if protected:
            if not hasattr(self._rust, "kv_lease_reclaim_foreign_except"):
                raise RuntimeError("gms_rust_ring lacks selective KV reclaim")
            return int(
                self._rust.kv_lease_reclaim_foreign_except(
                    self._mmap,
                    protected,
                    int(self._owner_hash),
                    max(0, int(max_blocks)),
                )
            )
        return int(
            self._rust.kv_lease_reclaim_foreign(
                self._mmap,
                int(self._owner_hash),
                max(0, int(max_blocks)),
            )
        )

    def raw_free_count(self) -> int:
        return int(self._rust.kv_lease_free_count(self._mmap))

    def free_count(self) -> int:
        return self.raw_free_count()

    def refresh_free_count(self) -> int:
        return self.free_count()


@dataclass(frozen=True)
class KVLeaseReclaimResult:
    files: int = 0
    reclaimed_blocks: int = 0
    errors: int = 0


def _owner_id_from_env(engine: str, device: int) -> str:
    engine_upper = engine.upper().replace("-", "_")
    return os.environ.get(
        f"GMS_{engine_upper}_KV_LEASE_OWNER_ID",
        os.environ.get("GMS_KV_LEASE_OWNER_ID", f"{engine}-{os.getpid()}-{device}"),
    )


def _kv_lease_shm_dir(engine: str) -> str:
    engine_upper = engine.upper().replace("-", "_")
    base_dir = os.environ.get(f"GMS_{engine_upper}_KV_LEASE_SHM_DIR")
    if base_dir is None:
        base_dir = os.environ.get("GMS_KV_LEASE_SHM_DIR")
    if not base_dir:
        base_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    return base_dir


def reclaim_foreign_kv_leases_in_shm_dir(
    engine: str,
    device: int,
    *,
    owner_id: str | None = None,
    shm_dir: str | None = None,
    max_blocks_per_file: int = 0,
    protected_blocks: set[int] | None = None,
    namespace_suffix: str = "kv",
) -> KVLeaseReclaimResult:
    """Reclaim non-self KV leases from THIS engine+device's lease mmap file.

    This is intended for post-fence hard failover only. Foreign owners in the
    caller's own lease namespace are orphaned primary processes after the shadow
    has acquired the failover lock. Call it before the replacement owner starts
    lease mutations; it also recovers interrupted transitions.

    Scope: only the lease file for this engine+device's namespace is touched. It
    must NOT glob every ``gms-kv-lease-*.shm`` in the (node-shared) directory --
    doing so reclaims LIVE leases belonging to healthy ranks/devices that share
    the directory, corrupting their KV. Reclaim is derived from the same
    namespace rule used by the client (``GMS_<ENGINE>_KV_LEASE_NAMESPACE`` ->
    ``GMS_KV_LEASE_NAMESPACE`` -> ``{engine}:gpu{device}:{namespace_suffix}``).

    ``protected_blocks`` are READY HBM slots from the authoritative content
    directory. They remain sealed for lazy adoption by the replacement owner;
    all other foreign records are reclaimed as before.
    """

    rust = _load_optional_rust_ring()
    if rust is None or not hasattr(rust, "kv_lease_reclaim_foreign"):
        return KVLeaseReclaimResult()

    protected = sorted(int(block_id) for block_id in (protected_blocks or set()))
    if protected and not hasattr(rust, "kv_lease_reclaim_foreign_except"):
        logger.error(
            "GMS KV selective reclaim unavailable; preserving all foreign "
            "leases rather than discarding directory-owned HBM"
        )
        return KVLeaseReclaimResult(errors=1)

    owner = owner_id or _owner_id_from_env(engine, device)
    owner_hash = _owner_hash(owner)
    # Resolve THIS engine+device's own lease file only (never a directory glob).
    engine_upper = engine.upper().replace("-", "_")
    namespace = os.environ.get(
        f"GMS_{engine_upper}_KV_LEASE_NAMESPACE",
        os.environ.get(
            "GMS_KV_LEASE_NAMESPACE",
            f"{engine}:gpu{device}:{namespace_suffix}",
        ),
    )
    explicit_path = os.environ.get(
        f"GMS_{engine_upper}_KV_LEASE_SHM_PATH"
    ) or os.environ.get("GMS_KV_LEASE_SHM_PATH")
    if explicit_path:
        target_paths = [Path(explicit_path)]
    else:
        base_dir = Path(shm_dir or _kv_lease_shm_dir(engine))
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:20]
        target_paths = [base_dir / f"gms-kv-lease-{digest}.shm"]
    files = 0
    reclaimed = 0
    errors = 0
    for path in target_paths:
        try:
            fd = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            continue
        try:
            header = _read_shm_header(fd)
            if not _valid_shm_header(header):
                continue
            assert header is not None
            map_size = (
                _KV_LEASE_SHM_HEADER_SIZE + int(header[2]) * _KV_LEASE_SHM_RECORD_SIZE
            )
            buf = mmap.mmap(fd, map_size)
            try:
                if protected:
                    n = int(
                        rust.kv_lease_reclaim_foreign_except(
                            buf,
                            protected,
                            int(owner_hash),
                            max(0, int(max_blocks_per_file)),
                        )
                    )
                else:
                    n = int(
                        rust.kv_lease_reclaim_foreign(
                            buf,
                            int(owner_hash),
                            max(0, int(max_blocks_per_file)),
                        )
                    )
            finally:
                buf.close()
            files += 1
            reclaimed += n
        except Exception:
            errors += 1
            # Reclaim failure during failover strands the ex-primary's leases,
            # permanently leaking HBM; surface it at WARNING so an operator can
            # see why capacity was lost rather than only in debug logs.
            logger.warning(
                "GMS KV failover foreign-lease reclaim failed for %s",
                path,
                exc_info=True,
            )
        finally:
            os.close(fd)
    return KVLeaseReclaimResult(
        files=files,
        reclaimed_blocks=reclaimed,
        errors=errors,
    )


def _load_optional_rust_ring():
    try:
        import gms_rust_ring  # type: ignore[import-not-found]
    except Exception:
        return None
    return gms_rust_ring


def _read_existing_total_blocks(shm_path: str) -> int | None:
    try:
        fd = os.open(shm_path, os.O_RDONLY)
    except FileNotFoundError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        header = _read_shm_header(fd)
        if not _valid_shm_header(header):
            return None
        assert header is not None
        return int(header[2])
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _init_or_adopt_kv_lease_namespace(
    *,
    shm_path: str,
    total_blocks: int,
    reserved_blocks: list[int] | None,
) -> int:
    """Initialize a KV lease namespace or adopt the already-created geometry.

    Engine sizing hooks call this before constructing their local KV allocator.
    During Bulwark startup, primary and shadow can race with slightly different
    candidate cache sizes for the same GPU namespace. The namespace creator
    wins; later contenders must adopt that size so all processes address the
    same lease map. The real SharedMemoryKVLeaseClient remains strict so
    post-sizing mismatches are still caught.
    """
    if total_blocks <= 0:
        raise ValueError("total_blocks must be positive")

    os.makedirs(os.path.dirname(os.path.abspath(shm_path)), exist_ok=True)
    fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    buf: mmap.mmap | None = None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        header = _read_shm_header(fd)
        stat = os.fstat(fd)
        should_init = (
            SharedMemoryKVLeaseClient._reset_requested()
            or not _valid_shm_header(header)
        )
        if should_init:
            map_blocks = int(total_blocks)
            map_size = (
                _KV_LEASE_SHM_HEADER_SIZE + map_blocks * _KV_LEASE_SHM_RECORD_SIZE
            )
            os.ftruncate(fd, map_size)
            buf = mmap.mmap(fd, map_size)
            SharedMemoryKVLeaseClient._load_rust_ring().kv_lease_init(
                buf, map_blocks, reserved_blocks or []
            )
            buf.flush()
            return map_blocks

        assert header is not None
        _magic, _version, existing_blocks, record_size, _free_count, _epoch = header
        if record_size != _KV_LEASE_SHM_RECORD_SIZE:
            raise RuntimeError(
                f"unsupported KV lease record size in {shm_path}: {record_size}"
            )
        map_size = (
            _KV_LEASE_SHM_HEADER_SIZE + int(existing_blocks) * _KV_LEASE_SHM_RECORD_SIZE
        )
        if stat.st_size < map_size:
            os.ftruncate(fd, map_size)
        if int(existing_blocks) != int(total_blocks):
            logger.warning(
                "Adopting existing KV lease shared-memory geometry: "
                "path=%s existing=%s requested=%s",
                shm_path,
                existing_blocks,
                total_blocks,
            )
        return int(existing_blocks)
    finally:
        if buf is not None:
            buf.close()
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def resolve_kv_lease_namespace_total_blocks(
    engine: str,
    device: int,
    *,
    total_blocks: int,
    namespace_suffix: str = "kv",
    reserved_blocks: list[int] | None = None,
    timeout_ms: int | None = None,
) -> tuple[str, int]:
    """Create or read a shared-memory KV lease namespace size."""
    _ = timeout_ms
    engine_upper = engine.upper().replace("-", "_")
    namespace = os.environ.get(
        f"GMS_{engine_upper}_KV_LEASE_NAMESPACE",
        os.environ.get(
            "GMS_KV_LEASE_NAMESPACE",
            f"{engine}:gpu{device}:{namespace_suffix}",
        ),
    )
    shm_path = _kv_lease_shm_path(engine, namespace)
    resolved_blocks = _init_or_adopt_kv_lease_namespace(
        shm_path=shm_path,
        total_blocks=total_blocks,
        reserved_blocks=reserved_blocks,
    )
    return namespace, resolved_blocks


def read_kv_lease_namespace_total_blocks(
    engine: str,
    device: int,
    *,
    namespace_suffix: str = "kv",
) -> tuple[str, int | None]:
    """Read an existing shared-memory KV lease namespace size.

    This is intentionally read-only: unlike
    ``resolve_kv_lease_namespace_total_blocks`` it never creates or resets the
    namespace. Engine attach paths use this before KV cache sizing so a shadow
    or restarted engine can adopt the already-created pool geometry instead of
    deriving a smaller cache from currently free HBM.
    """
    engine_upper = engine.upper().replace("-", "_")
    namespace = os.environ.get(
        f"GMS_{engine_upper}_KV_LEASE_NAMESPACE",
        os.environ.get(
            "GMS_KV_LEASE_NAMESPACE",
            f"{engine}:gpu{device}:{namespace_suffix}",
        ),
    )
    shm_path = _kv_lease_shm_path(engine, namespace)
    return namespace, _read_existing_total_blocks(shm_path)


# Compatibility name used by engine integrations. It is intentionally backed by
# the Rust shared-memory implementation, not the removed per-allocation RPC path.
GMSKVLeaseClient = SharedMemoryKVLeaseClient


def leases_by_block_id(leases: list[KVLease]) -> dict[int, KVLease]:
    return {lease.block_id: lease for lease in leases}


def kv_leases_enabled(engine: str) -> bool:
    engine_upper = engine.upper().replace("-", "_")
    value = os.environ.get(f"GMS_{engine_upper}_KV_LEASES")
    if value is None:
        value = os.environ.get("GMS_KV_LEASES", "0")
    return value.strip().lower() not in ("", "0", "false", "no", "off")
