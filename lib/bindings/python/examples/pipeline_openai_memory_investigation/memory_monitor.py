"""
Shared memory monitoring utilities for Dynamo components.

Environment variables:
  DYNAMO_MEMORY_PROFILE=1  - Enable basic monitoring (default: enabled)
  DYNAMO_MEMORY_TRACE=1    - Enable tracemalloc allocation tracking
  DYNAMO_MEMORY_VERBOSE=1  - Enable object snapshots and growth tracking
  DYNAMO_MEMORY_DEBUG=1    - Enable per-request lifecycle tracking (high overhead)
"""
import asyncio
import ctypes
import gc
import os
import sys
import threading
import time
import tracemalloc
from typing import Optional

import psutil

# Load libc for malloc_trim and malloc_stats
try:
    _libc = ctypes.CDLL("libc.so.6")
    _malloc_trim = _libc.malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int

    _malloc_stats = _libc.malloc_stats
    _malloc_stats.argtypes = []
    _malloc_stats.restype = None

    # malloc_info writes XML to a FILE*
    _malloc_info = _libc.malloc_info
    _malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
    _malloc_info.restype = ctypes.c_int

    HAS_MALLOC_TRIM = True
    HAS_MALLOC_STATS = True
except (OSError, AttributeError):
    HAS_MALLOC_TRIM = False
    HAS_MALLOC_STATS = False
    _malloc_trim = None
    _malloc_stats = None
    _malloc_info = None


# mallinfo2 structure for detailed arena info (glibc 2.33+)
class MallInfo2(ctypes.Structure):
    """Structure returned by mallinfo2() - sizes in bytes"""
    _fields_ = [
        ("arena", ctypes.c_size_t),      # Non-mmapped space allocated (bytes)
        ("ordblks", ctypes.c_size_t),    # Number of free chunks
        ("smblks", ctypes.c_size_t),     # Number of free fastbin blocks
        ("hblks", ctypes.c_size_t),      # Number of mmapped regions
        ("hblkhd", ctypes.c_size_t),     # Space allocated in mmapped regions (bytes)
        ("usmblks", ctypes.c_size_t),    # See below
        ("fsmblks", ctypes.c_size_t),    # Space in freed fastbin blocks (bytes)
        ("uordblks", ctypes.c_size_t),   # Total allocated space (bytes)
        ("fordblks", ctypes.c_size_t),   # Total free space (bytes)
        ("keepcost", ctypes.c_size_t),   # Top-most, releasable space (bytes)
    ]


try:
    _mallinfo2 = _libc.mallinfo2
    _mallinfo2.argtypes = []
    _mallinfo2.restype = MallInfo2
    HAS_MALLINFO2 = True
except (OSError, AttributeError):
    HAS_MALLINFO2 = False
    _mallinfo2 = None

# Optional objgraph for reference chain visualization
try:
    import objgraph

    HAS_OBJGRAPH = True
except ImportError:
    HAS_OBJGRAPH = False

# PyO3 classes that are suspects for memory leaks
PYCLASS_SUSPECTS = [
    "Context",
    "AsyncResponseStream",
    "Client",
    "HttpAsyncEngine",
    "PythonAsyncEngine",
    "NvChatCompletion",
    "NvCompletionChoice",
    "Endpoint",
    "Component",
]


# -------------------------------------------------------------------------
# Helper Functions for Object Size Estimation
# -------------------------------------------------------------------------


def estimate_pyobject_size(obj) -> int:
    """Recursively estimate Python object size including nested objects.

    This provides a rough estimate of the total memory footprint of a Python
    object by traversing its contents. Useful for measuring serialized request
    sizes.
    """
    seen: set[int] = set()

    def sizeof(o) -> int:
        obj_id = id(o)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        try:
            size = sys.getsizeof(o)
        except (TypeError, RecursionError):
            return 0

        if isinstance(o, dict):
            size += sum(sizeof(k) + sizeof(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set, frozenset)):
            size += sum(sizeof(i) for i in o)
        elif hasattr(o, "__dict__"):
            size += sizeof(o.__dict__)
        elif hasattr(o, "__slots__"):
            size += sum(sizeof(getattr(o, slot, None)) for slot in o.__slots__ if hasattr(o, slot))

        return size

    return sizeof(obj)


def get_rss_mb() -> float:
    """Get current RSS memory in MB"""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# -------------------------------------------------------------------------
# glibc Arena Debugging Functions
# -------------------------------------------------------------------------


def dump_malloc_stats():
    """Print glibc malloc statistics to stderr.

    This shows per-arena breakdown of:
    - system bytes: memory obtained from OS
    - in use bytes: memory currently allocated
    - Total (incl. mmap)

    See: https://man7.org/linux/man-pages/man3/malloc_stats.3.html
    """
    if HAS_MALLOC_STATS:
        print("\n=== glibc malloc_stats() ===", flush=True)
        _malloc_stats()  # Prints to stderr
        print("=== end malloc_stats ===\n", flush=True)
    else:
        print("malloc_stats not available")


def get_mallinfo2() -> dict:
    """Get detailed malloc info using mallinfo2().

    Returns dict with:
    - arena: Non-mmapped space allocated (bytes)
    - ordblks: Number of free chunks
    - hblks: Number of mmapped regions
    - hblkhd: Space in mmapped regions (bytes)
    - uordblks: Total allocated space (bytes)
    - fordblks: Total free space (bytes)
    - keepcost: Releasable space at top of heap (bytes)

    See: https://man7.org/linux/man-pages/man3/mallinfo.3.html
    """
    if not HAS_MALLINFO2:
        return {}

    info = _mallinfo2()
    return {
        "arena_mb": info.arena / 1024 / 1024,
        "free_chunks": info.ordblks,
        "mmap_regions": info.hblks,
        "mmap_mb": info.hblkhd / 1024 / 1024,
        "allocated_mb": info.uordblks / 1024 / 1024,
        "free_mb": info.fordblks / 1024 / 1024,
        "releasable_mb": info.keepcost / 1024 / 1024,
    }


def dump_arena_info(label: str = ""):
    """Print arena allocation summary.

    This helps diagnose glibc arena fragmentation issues.
    High 'arena' with low 'allocated' indicates fragmentation.
    High 'free_mb' with low 'releasable_mb' indicates retained memory.
    """
    if not HAS_MALLINFO2:
        print("mallinfo2 not available (requires glibc 2.33+)")
        return

    info = get_mallinfo2()
    prefix = f"[{label}] " if label else ""

    print(f"{prefix}=== Arena Info ===")
    print(f"{prefix}  Arena (non-mmap): {info['arena_mb']:.1f} MB")
    print(f"{prefix}  Allocated:        {info['allocated_mb']:.1f} MB")
    print(f"{prefix}  Free in arenas:   {info['free_mb']:.1f} MB")
    print(f"{prefix}  Releasable:       {info['releasable_mb']:.1f} MB")
    print(f"{prefix}  Mmap regions:     {info['mmap_regions']} ({info['mmap_mb']:.1f} MB)")
    print(f"{prefix}  Free chunks:      {info['free_chunks']}")

    # Calculate fragmentation ratio
    if info['arena_mb'] > 0:
        utilization = info['allocated_mb'] / info['arena_mb'] * 100
        print(f"{prefix}  Arena utilization: {utilization:.1f}%")
        if utilization < 50:
            print(f"{prefix}  WARNING: Low utilization suggests arena fragmentation!")

    # Check if memory could be returned
    if info['free_mb'] > 10 and info['releasable_mb'] < info['free_mb'] * 0.1:
        print(f"{prefix}  WARNING: {info['free_mb']:.1f}MB free but only {info['releasable_mb']:.1f}MB releasable!")
        print(f"{prefix}  This indicates memory fragmentation - free chunks not at heap top.")

    print(f"{prefix}==================", flush=True)


# -------------------------------------------------------------------------
# Request Memory Tracker for Per-Request Analysis
# -------------------------------------------------------------------------


class RequestMemoryTracker:
    """Track memory consumption per request to identify payload-correlated leaks.

    Usage:
        tracker = RequestMemoryTracker()
        # For each request:
        tracker.record(payload_size=200000, mem_delta_kb=500.0)
        # Periodically:
        tracker.report()
    """

    def __init__(self, max_samples: int = 10000):
        self.deltas: list[float] = []
        self.payload_sizes: list[int] = []
        self.max_samples = max_samples

    def record(self, payload_size: int, mem_delta_kb: float):
        """Record a request's payload size and memory delta."""
        self.deltas.append(mem_delta_kb)
        self.payload_sizes.append(payload_size)

        # Keep bounded history
        if len(self.deltas) > self.max_samples:
            self.deltas = self.deltas[-self.max_samples:]
            self.payload_sizes = self.payload_sizes[-self.max_samples:]

    def report(self, component_name: str = ""):
        """Print analysis of memory retention vs payload size."""
        if not self.deltas:
            return

        prefix = f"[{component_name}] " if component_name else ""

        # Calculate bytes retained per byte of payload
        retention_ratios: list[float] = []
        for payload, delta in zip(self.payload_sizes, self.deltas):
            if payload > 0:
                retention_ratios.append((delta * 1024) / payload)

        if retention_ratios:
            avg_ratio = sum(retention_ratios) / len(retention_ratios)
            max_ratio = max(retention_ratios)
            min_ratio = min(retention_ratios)
            print(
                f"{prefix}Memory retention ratio: avg={avg_ratio:.2f}x, min={min_ratio:.2f}x, max={max_ratio:.2f}x payload size"
            )

        # Group by payload size to see correlation
        size_buckets: dict[int, list[float]] = {}
        for payload, delta in zip(self.payload_sizes, self.deltas):
            bucket = payload // 10000 * 10000  # Round to nearest 10KB
            if bucket not in size_buckets:
                size_buckets[bucket] = []
            size_buckets[bucket].append(delta)

        print(f"{prefix}Memory delta by payload size:")
        for bucket in sorted(size_buckets.keys()):
            deltas = size_buckets[bucket]
            avg = sum(deltas) / len(deltas)
            print(f"    {bucket/1000:.0f}KB payload: avg delta={avg:.1f}KB ({len(deltas)} samples)")

    def summary_stats(self) -> dict:
        """Return summary statistics."""
        if not self.deltas:
            return {}

        return {
            "num_samples": len(self.deltas),
            "avg_delta_kb": sum(self.deltas) / len(self.deltas),
            "max_delta_kb": max(self.deltas),
            "min_delta_kb": min(self.deltas),
            "total_delta_mb": sum(self.deltas) / 1024,
        }


class MemoryMonitor:
    """Memory monitor that can be shared across components"""

    def __init__(
        self,
        component_name: str,
        enable_trace: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.process = psutil.Process(os.getpid())
        self.request_count = 0
        self.start_time = time.time()
        self.component_name = component_name
        self.freed_last = 0.0
        self.verbose = verbose
        self.debug = debug

        # Object snapshot tracking
        self._last_snapshot: Optional[dict[str, int]] = None

        # Threshold tracking for dumps (to handle concurrent increments)
        self._last_log_threshold = 0
        self._last_object_dump_threshold = 0
        self._last_gc_dump_threshold = 0
        self._last_refchain_threshold = 0

        # Tracemalloc
        self._tracemalloc_enabled = enable_trace
        self._tracemalloc_snapshot = None
        if enable_trace:
            self.start_tracemalloc()

        # Per-request tracking (debug mode)
        self.request_tracker = RequestMemoryTracker() if debug else None
        self._active_requests: dict[int, tuple[float, int]] = {}  # request_id -> (start_rss_mb, payload_size)

    @property
    def initial_memory(self):
        if not hasattr(self, "_initial_memory"):
            self._initial_memory = self.process.memory_info().rss
        return self._initial_memory

    # -------------------------------------------------------------------------
    # Basic Memory Logging
    # -------------------------------------------------------------------------

    def log_memory(self, label: str = ""):
        """Log current memory usage"""
        memory_info = self.process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        rss_growth = (memory_info.rss - self.initial_memory) / 1024 / 1024
        uptime = time.time() - self.start_time

        print(
            f"[{self.component_name}] [{uptime:.1f}s] {label} Memory: RSS={rss_mb:.2f}MB (+{rss_growth:.2f}MB), VMS={vms_mb:.2f}MB, Requests={self.request_count}, Last GC freed={self.freed_last:.2f}MB"
        )

    # -------------------------------------------------------------------------
    # Object Snapshot & Growth Tracking (gc.get_objects)
    # -------------------------------------------------------------------------

    def snapshot_objects(self) -> dict[str, int]:
        """Snapshot current object counts by type"""
        counts: dict[str, int] = {}
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def diff_snapshots(
        self, before: dict[str, int], after: dict[str, int], top_n: int = 20
    ) -> list[tuple[str, int, int]]:
        """Return types with largest growth: (type_name, delta, current_count)"""
        growth: list[tuple[str, int, int]] = []
        for type_name, count in after.items():
            delta = count - before.get(type_name, 0)
            if delta > 0:
                growth.append((type_name, delta, count))
        return sorted(growth, key=lambda x: -x[1])[:top_n]

    def dump_object_growth(self):
        """Dump object growth since last snapshot"""
        current = self.snapshot_objects()
        if self._last_snapshot:
            growth = self.diff_snapshots(self._last_snapshot, current)
            if growth:
                print(f"[{self.component_name}] Object growth since last snapshot:", flush=True)
                for type_name, delta, total in growth[:15]:
                    print(f"    {type_name}: +{delta} (total: {total})", flush=True)
            else:
                print(f"[{self.component_name}] No significant object growth", flush=True)
        else:
            print(f"[{self.component_name}] First snapshot taken ({len(current)} types)", flush=True)
        self._last_snapshot = current

    def dump_top_objects(self, top_n: int = 25):
        """Dump top objects by count (all objects, not just growth)"""
        current = self.snapshot_objects()
        sorted_types = sorted(current.items(), key=lambda x: -x[1])[:top_n]
        print(f"[{self.component_name}] Top {top_n} object types by count:", flush=True)
        for type_name, count in sorted_types:
            print(f"    {type_name}: {count}", flush=True)

    def dump_all_objects_detailed(self, top_n: int = 30):
        """Dump detailed object info including module for top types"""
        # Count by (module, type_name)
        counts: dict[tuple[str, str], int] = {}
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            module = type(obj).__module__
            key = (module, type_name)
            counts[key] = counts.get(key, 0) + 1

        sorted_types = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
        print(f"[{self.component_name}] Top {top_n} object types (module.type: count):", flush=True)
        for (module, type_name), count in sorted_types:
            print(f"    {module}.{type_name}: {count}", flush=True)

    # -------------------------------------------------------------------------
    # PyO3 Object Tracking
    # -------------------------------------------------------------------------

    def count_pyo3_objects(self) -> dict[str, int]:
        """Count suspected PyO3 objects (only from dynamo._core module)"""
        counts = {name: 0 for name in PYCLASS_SUSPECTS}
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            # Only count objects from dynamo._core, not Python builtins like _contextvars.Context
            if type_name in counts and type(obj).__module__ == "dynamo._core":
                counts[type_name] += 1
        return {k: v for k, v in counts.items() if v > 0}

    def dump_pyo3_objects(self):
        """Dump counts of suspected PyO3 objects"""
        pyo3_counts = self.count_pyo3_objects()
        if pyo3_counts:
            print(f"[{self.component_name}] PyO3 objects: {pyo3_counts}", flush=True)
            # Debug: Show details about Context objects
            if "Context" in pyo3_counts and pyo3_counts["Context"] > 0:
                contexts = [obj for obj in gc.get_objects() if type(obj).__name__ == "Context"]
                if contexts:
                    print(f"[{self.component_name}] Context details (first 3):", flush=True)
                    for i, ctx in enumerate(contexts[:3]):
                        module = type(ctx).__module__
                        print(f"    [{i}] module={module}, type={type(ctx)}", flush=True)
        else:
            print(f"[{self.component_name}] No PyO3 suspect objects found", flush=True)

    # -------------------------------------------------------------------------
    # Uncollectable Objects (gc.garbage)
    # -------------------------------------------------------------------------

    def dump_uncollectable(self):
        """Dump objects in gc.garbage (have __del__ + in reference cycles)"""
        gc.collect()
        if gc.garbage:
            print(f"[{self.component_name}] UNCOLLECTABLE OBJECTS: {len(gc.garbage)}", flush=True)
            for i, obj in enumerate(gc.garbage[:10]):
                try:
                    obj_repr = repr(obj)[:100]
                except Exception:
                    obj_repr = "<repr failed>"
                print(f"    [{i}] {type(obj).__name__}: {obj_repr}", flush=True)
        else:
            print(f"[{self.component_name}] No uncollectable objects in gc.garbage", flush=True)

    # -------------------------------------------------------------------------
    # Tracemalloc Integration
    # -------------------------------------------------------------------------

    def start_tracemalloc(self, nframes: int = 25):
        """Start memory allocation tracking"""
        if not tracemalloc.is_tracing():
            tracemalloc.start(nframes)
            print(f"[{self.component_name}] tracemalloc started with {nframes} frames")
        self._tracemalloc_enabled = True

    def take_tracemalloc_snapshot(self):
        """Take a snapshot for comparison"""
        if self._tracemalloc_enabled:
            self._tracemalloc_snapshot = tracemalloc.take_snapshot()

    def diff_tracemalloc(self, top_n: int = 10):
        """Compare current allocations to last snapshot"""
        if not self._tracemalloc_enabled:
            return

        current = tracemalloc.take_snapshot()

        if self._tracemalloc_snapshot:
            diff = current.compare_to(self._tracemalloc_snapshot, "lineno")
            print(f"[{self.component_name}] Top {top_n} memory allocation changes:", flush=True)
            for stat in diff[:top_n]:
                print(f"    {stat}", flush=True)
        else:
            # First time - just show top allocations
            stats = current.statistics("lineno")
            print(f"[{self.component_name}] Top {top_n} memory allocations:", flush=True)
            for stat in stats[:top_n]:
                print(f"    {stat}", flush=True)

        self._tracemalloc_snapshot = current

    # -------------------------------------------------------------------------
    # Objgraph Integration (reference chain visualization)
    # -------------------------------------------------------------------------

    def show_object_growth(self, limit: int = 10):
        """Show which types grew the most since last call (requires objgraph)"""
        if not HAS_OBJGRAPH:
            print(f"[{self.component_name}] objgraph not installed, skipping show_growth")
            return
        print(f"[{self.component_name}] Object growth (objgraph):")
        objgraph.show_growth(limit=limit)

    def find_leaking_objects(self, type_name: str, sample_count: int = 3):
        """Find objects of given type and show what's holding them"""
        if not HAS_OBJGRAPH:
            print(f"[{self.component_name}] objgraph not installed")
            return

        objects = objgraph.by_type(type_name)
        print(f"[{self.component_name}] Found {len(objects)} '{type_name}' objects")

        if objects and sample_count > 0:
            for i, obj in enumerate(objects[:sample_count]):
                print(f"    [{i}] Backrefs for {type_name}:")
                # Get text representation of backrefs
                chain = objgraph.find_backref_chain(obj, objgraph.is_proper_module, max_depth=5)
                if chain:
                    print(f"        Chain length: {len(chain)}")
                    for j, ref in enumerate(chain[:5]):
                        print(f"        -> {type(ref).__name__}: {repr(ref)[:60]}")

    def dump_reference_chains(self, suspect_types: Optional[list[str]] = None, threshold: int = 1000):
        """Dump reference chains for suspected leaking types"""
        if not HAS_OBJGRAPH:
            return

        suspects = suspect_types or ["Context", "AsyncResponseStream", "dict"]
        for type_name in suspects:
            count = objgraph.count(type_name)
            if count > threshold:
                print(
                    f"[{self.component_name}] {type_name}: {count} objects (>{threshold}) - investigating..."
                )
                self.find_leaking_objects(type_name, sample_count=2)

    # -------------------------------------------------------------------------
    # Request Lifecycle Tracking (debug mode)
    # -------------------------------------------------------------------------

    def request_start(self, request_id: int, request: dict) -> None:
        """Call at the start of a request to track memory delta.

        Args:
            request_id: Unique identifier for this request (use id(request))
            request: The request dict to measure size of
        """
        if not self.debug:
            return

        rss_mb = get_rss_mb()
        payload_size = estimate_pyobject_size(request)
        self._active_requests[request_id] = (rss_mb, payload_size)

        refcount = sys.getrefcount(request)
        print(
            f"[{self.component_name}] [REQ {request_id}] START: "
            f"payload={payload_size}B, refcount={refcount}, RSS={rss_mb:.1f}MB",
            flush=True,
        )

    def request_end(self, request_id: int) -> None:
        """Call at the end of a request to measure memory delta.

        Args:
            request_id: Same identifier used in request_start
        """
        if not self.debug or request_id not in self._active_requests:
            return

        start_rss_mb, payload_size = self._active_requests.pop(request_id)
        gc.collect()
        end_rss_mb = get_rss_mb()
        delta_mb = end_rss_mb - start_rss_mb

        print(
            f"[{self.component_name}] [REQ {request_id}] END: "
            f"RSS={end_rss_mb:.1f}MB (delta: {delta_mb:+.2f}MB)",
            flush=True,
        )

        # Record for analysis
        if self.request_tracker:
            self.request_tracker.record(payload_size, delta_mb * 1024)  # Convert to KB

    def report_request_stats(self) -> None:
        """Print summary of per-request memory tracking."""
        if self.request_tracker:
            print(f"\n[{self.component_name}] === Per-Request Memory Analysis ===", flush=True)
            self.request_tracker.report(self.component_name)
            stats = self.request_tracker.summary_stats()
            if stats:
                print(
                    f"[{self.component_name}] Summary: {stats['num_samples']} requests, "
                    f"avg delta={stats['avg_delta_kb']:.1f}KB, "
                    f"total={stats['total_delta_mb']:.1f}MB",
                    flush=True,
                )

    def track_generator_scope(self, gen, request_id: int) -> None:
        """Log generator scope variables (call on first yield).

        Args:
            gen: The async generator object
            request_id: Request identifier for logging
        """
        if not self.debug:
            return

        try:
            frame = getattr(gen, "ag_frame", None) or getattr(gen, "gi_frame", None)
            if frame:
                locals_keys = list(frame.f_locals.keys())
                locals_size = sum(
                    sys.getsizeof(v)
                    for v in frame.f_locals.values()
                    if v is not None
                )
                print(
                    f"[{self.component_name}] [GEN {request_id}] Scope: "
                    f"size={locals_size}B, vars={locals_keys}",
                    flush=True,
                )
        except Exception as e:
            print(f"[{self.component_name}] [GEN {request_id}] Scope inspection failed: {e}", flush=True)

    def comprehensive_memory_report(self) -> None:
        """Print a comprehensive memory analysis report."""
        gc.collect()

        print(f"\n{'='*60}")
        print(f"[{self.component_name}] MEMORY REPORT @ {self.request_count} requests")
        print(f"{'='*60}")

        # RSS
        rss = get_rss_mb()
        initial_mb = self.initial_memory / 1024 / 1024
        print(f"RSS: {rss:.1f}MB (delta: +{rss - initial_mb:.1f}MB from start)")

        # Object counts by type (top 20)
        type_counts: dict[str, int] = {}
        for obj in gc.get_objects():
            t = type(obj).__name__
            type_counts[t] = type_counts.get(t, 0) + 1

        print(f"\nTop 20 object types:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {t}: {count:,}")

        # Suspect PyO3 types
        pyo3_counts = self.count_pyo3_objects()
        print(f"\nPyO3 suspects: {pyo3_counts}")

        # Uncollectable
        if gc.garbage:
            print(f"\nUNCOLLECTABLE: {len(gc.garbage)} objects")
            for obj in gc.garbage[:5]:
                try:
                    print(f"  {type(obj).__name__}: {repr(obj)[:80]}")
                except Exception:
                    print(f"  {type(obj).__name__}: <repr failed>")
        else:
            print(f"\nNo uncollectable objects")

        # Per-request stats if available
        if self.request_tracker and self.request_tracker.deltas:
            print(f"\nPer-request memory stats:")
            self.request_tracker.report(self.component_name)

        print(f"{'='*60}\n")

    # -------------------------------------------------------------------------
    # Combined Dump Methods
    # -------------------------------------------------------------------------

    def dump_full_report(self):
        """Dump a comprehensive memory report"""
        print(f"\n[{self.component_name}] ========== FULL MEMORY REPORT ==========")
        self.log_memory("Current state:")
        print()
        self.dump_object_growth()
        print()
        self.dump_pyo3_objects()
        print()
        self.dump_uncollectable()
        print()
        if self._tracemalloc_enabled:
            self.diff_tracemalloc()
            print()
        if HAS_OBJGRAPH:
            self.show_object_growth()
        print(f"[{self.component_name}] ========================================\n")

    # -------------------------------------------------------------------------
    # Request Tracking (enhanced)
    # -------------------------------------------------------------------------

    def increment_request(self):
        """Increment request counter and trigger periodic dumps"""
        self.request_count += 1
        self.initial_memory  # Ensure initial memory is captured
        count = self.request_count  # Capture for consistent checks

        # Every 5000 requests - log basic memory
        current_log_threshold = count // 5000
        if current_log_threshold > self._last_log_threshold:
            self._last_log_threshold = current_log_threshold
            threading.Thread(
                target=self.log_memory,
                args=(f"After {count} requests:",),
                daemon=True,
            ).start()

        # Every 10000 requests - dump object growth (if verbose)
        current_object_threshold = count // 10000
        if self.verbose and current_object_threshold > self._last_object_dump_threshold:
            self._last_object_dump_threshold = current_object_threshold
            print(f"[{self.component_name}] === Object Growth Dump (at {count} requests) ===", flush=True)
            self.dump_object_growth()
            self.dump_pyo3_objects()

        # Every 20000 requests - full GC + malloc_trim + detailed analysis
        current_gc_threshold = count // 20000
        if current_gc_threshold > self._last_gc_dump_threshold:
            self._last_gc_dump_threshold = current_gc_threshold
            print(f"[{self.component_name}] === GC + malloc_trim + Analysis (at {count} requests) ===", flush=True)
            before_gc = self.process.memory_info().rss
            collected = gc.collect()
            after_gc = self.process.memory_info().rss

            # Call malloc_trim to release memory back to OS
            # Controlled by DYNAMO_MALLOC_TRIM env var (default: disabled for A/B testing)
            trim_result = 0
            if HAS_MALLOC_TRIM and os.getenv("DYNAMO_MALLOC_TRIM", "0") == "1":
                trim_result = _malloc_trim(0)
            after_trim = self.process.memory_info().rss

            gc_freed_mb = (before_gc - after_gc) / 1024 / 1024
            trim_freed_mb = (after_gc - after_trim) / 1024 / 1024
            total_freed_mb = (before_gc - after_trim) / 1024 / 1024
            self.freed_last = total_freed_mb

            print(
                f"[{self.component_name}]   GC collected {collected} objects, freed {gc_freed_mb:.2f}MB",
                flush=True,
            )
            print(
                f"[{self.component_name}]   malloc_trim returned {trim_result}, freed {trim_freed_mb:.2f}MB (total freed: {total_freed_mb:.2f}MB)",
                flush=True,
            )
            self.dump_uncollectable()
            # Dump all objects with module info
            self.dump_all_objects_detailed(top_n=30)

            if self._tracemalloc_enabled:
                self.diff_tracemalloc()

        # Every 50000 requests - investigate reference chains
        current_refchain_threshold = count // 50000
        if self.verbose and current_refchain_threshold > self._last_refchain_threshold:
            self._last_refchain_threshold = current_refchain_threshold
            print(f"[{self.component_name}] === Reference Chain Analysis (at {count} requests) ===", flush=True)
            self.dump_reference_chains()


def create_monitor(component_name: str) -> Optional[MemoryMonitor]:
    """Create a memory monitor based on environment variables.

    Environment variables:
      DYNAMO_MEMORY_PROFILE=1  - Enable basic monitoring (default: enabled)
      DYNAMO_MEMORY_TRACE=1    - Enable tracemalloc allocation tracking
      DYNAMO_MEMORY_VERBOSE=1  - Enable object snapshots and growth tracking
      DYNAMO_MEMORY_DEBUG=1    - Enable per-request lifecycle tracking (high overhead)
    """
    if os.getenv("DYNAMO_MEMORY_PROFILE", "1") != "1":
        return None

    enable_trace = os.getenv("DYNAMO_MEMORY_TRACE", "0") == "1"
    verbose = os.getenv("DYNAMO_MEMORY_VERBOSE", "0") == "1"
    debug = os.getenv("DYNAMO_MEMORY_DEBUG", "0") == "1"

    monitor = MemoryMonitor(
        component_name,
        enable_trace=enable_trace,
        verbose=verbose,
        debug=debug,
    )

    if enable_trace:
        print(f"[{component_name}] Memory tracing enabled (tracemalloc)")
    if verbose:
        print(f"[{component_name}] Verbose object tracking enabled")
    if debug:
        print(f"[{component_name}] Debug mode enabled (per-request tracking - HIGH OVERHEAD)")
    if not HAS_OBJGRAPH:
        print(f"[{component_name}] Note: Install objgraph for reference chain analysis")

    return monitor


def setup_background_monitor(monitor: Optional[MemoryMonitor]):
    """Create background monitoring task if monitor exists"""
    if not monitor:
        print("Memory monitoring not enabled.")
        return None

    async def background_monitor():
        while True:
            await asyncio.sleep(15)  # Log every 15 seconds
            monitor.log_memory("Background check:")

    return asyncio.create_task(background_monitor())
