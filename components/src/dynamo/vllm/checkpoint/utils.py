# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""General utilities for checkpoint/restore operations."""

import os
import time
import logging

logger = logging.getLogger(__name__)

# Try to import cuda-python
try:
    from cuda import cuda
    cuda_available = True
except Exception:  # pragma: no cover
    logger.warning("cuda-python package not found. CUDA checkpointing will not be available. "
                   "Install with: pip install cuda-python")
    cuda = None
    cuda_available = False


# General checkpoint utilities

def read_comm(pid: int) -> str:
    """Best-effort read of the process command name."""
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "?"


def read_cmdline(pid: int) -> str:
    """Best-effort read of the full command line."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        if not raw:
            return ""
        parts = [p.decode("utf-8", "ignore") for p in raw.split(b"\x00") if p]
        return " ".join(parts)
    except Exception:
        return ""


def collect_process_tree_pids(root_pid: int) -> set[int]:
    """Recursively collect PIDs in the process tree rooted at root_pid.

    Uses /proc/<pid>/task/<pid>/children to discover descendants.
    Best-effort: missing /proc entries are ignored.
    """
    pending: list[int] = [root_pid]
    seen: set[int] = set[int]()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        children_path = f"/proc/{pid}/task/{pid}/children"
        try:
            with open(children_path, encoding="utf-8") as f:
                content = f.read().strip()
        except FileNotFoundError:
            continue
        except Exception:
            continue
        if not content:
            continue
        for token in content.split():
            try:
                child = int(token)
            except ValueError:
                continue
            pending.append(child)
    return seen


def process_is_leaf(pid: int) -> bool:
    """Check if a process is a leaf (has no children).

    Args:
        pid: Process ID to check

    Returns:
        True if the process has no children, False otherwise
    """
    children_path = f"/proc/{pid}/task/{pid}/children"
    try:
        with open(children_path, encoding="utf-8") as f:
            content = f.read().strip()
        return not content  # Empty means no children
    except Exception:
        # If we can't read, assume it's a leaf to be safe
        return True


def verify_processes_exited(pre_dump_pids: set[int], timeout_s: float = 5.0) -> list[int]:
    """Verify that all processes from pre-dump snapshot have exited.

    Args:
        pre_dump_pids: Set of PIDs that existed before CRIU dump
        timeout_s: Maximum time to wait for processes to exit

    Returns:
        List of PIDs that are still running after timeout
    """
    deadline = time.time() + timeout_s
    lingering: set[int] = set()

    while time.time() < deadline:
        lingering = {
            pid for pid in pre_dump_pids
            if os.path.exists(f"/proc/{pid}")
        }
        if not lingering:
            break
        time.sleep(0.05)

    if lingering:
        logger.error(
            "CRIU dump verification: %d lingering PIDs: %s",
            len(lingering),
            sorted(lingering)
        )
        # Collect info about lingering processes
        for pid in sorted(lingering):
            try:
                comm = read_comm(pid)
                cmdline = read_cmdline(pid)
                logger.error("  PID %d (%s): %s", pid, comm, cmdline)
            except Exception:
                logger.error("  PID %d (unable to read info)", pid)
    else:
        logger.info("CRIU dump verification: all pre-dump PIDs have exited")

    return list(lingering)


def get_tty_info(pid: int) -> tuple[str, str]:
    """Get TTY device info for a process.

    Args:
        pid: Process ID to get TTY info for

    Returns:
        Tuple of (rdev, dev) as hex strings for CRIU
    """
    try:
        # Get the TTY device from /proc/PID/fd/0
        tty_path = f"/proc/{pid}/fd/0"
        st = os.stat(tty_path)

        # Format as hex values for CRIU
        rdev = f"{st.st_rdev:x}"
        dev = f"{st.st_dev:x}"

        return rdev, dev
    except Exception as e:
        logger.warning("Could not get TTY info: %s", e)
        return "", ""


# CUDA checkpoint utilities

def get_cuda_error_details(err) -> tuple[str, str]:
    """Get detailed error information from a CUDA error code.
    
    Returns:
        Tuple of (error_name, error_description)
    """
    # Get error name
    _, name_ptr = cuda.cuGetErrorName(err)
    error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
    
    # Get error description
    try:
        _, desc_ptr = cuda.cuGetErrorString(err)
        error_desc = desc_ptr.decode() if isinstance(desc_ptr, bytes) else str(desc_ptr)
    except Exception:
        error_desc = "No description available"
    
    return error_name, error_desc


def get_process_gpu_devices(pid: int) -> list[int]:
    """Get list of GPU device indices that a process has open file descriptors to.
    
    Args:
        pid: Process ID to check
        
    Returns:
        List of GPU device indices (e.g., [0, 1, 3])
    """
    fd_dir = f"/proc/{pid}/fd"
    devices = set()
    
    if not os.path.exists(fd_dir):
        return []
    
    try:
        for fd in os.listdir(fd_dir):
            try:
                link = os.readlink(os.path.join(fd_dir, fd))
                # Check for /dev/nvidia0, /dev/nvidia1, etc.
                if link.startswith("/dev/nvidia") and link[12:].isdigit():
                    device_idx = int(link[12:])
                    devices.add(device_idx)
            except Exception:
                continue
    except Exception:
        pass
    
    return sorted(list(devices))


def log_gpu_memory_for_devices(devices: list[int], context: str = "") -> None:
    """Log memory usage for specific GPU devices.
    
    Args:
        devices: List of GPU device indices to log
        context: Context string to include in log messages
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        
        for device_idx in devices:
            if device_idx < torch.cuda.device_count():
                try:
                    allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
                    reserved = torch.cuda.memory_reserved(device_idx) / 1024**3
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
                    free = free_bytes / 1024**3
                    total = total_bytes / 1024**3
                    props = torch.cuda.get_device_properties(device_idx)
                    
                    logger.info(
                        "[CUDA CHECKPOINT%s] GPU %d (%s): "
                        "Allocated=%.2fGB, Reserved=%.2fGB, Free=%.2fGB, Total=%.2fGB",
                        f" {context}" if context else "",
                        device_idx,
                        props.name,
                        allocated,
                        reserved,
                        free,
                        total
                    )
                except Exception as e:
                    logger.warning("Could not get memory info for GPU %d: %s", device_idx, e)
    except ImportError:
        pass


def checkpoint_cuda_process(pid: int) -> None:
    """Lock and checkpoint a CUDA process using the CUDA checkpoint API."""
    if not cuda_available:
        raise RuntimeError("cuda-python package not available")

    # Get which GPU devices this process is using
    gpu_devices = get_process_gpu_devices(pid)
    if gpu_devices:
        logger.info("PID %d is using GPU device(s): %s", pid, gpu_devices)
        log_gpu_memory_for_devices(gpu_devices, f"PID_{pid}_BEFORE")
    else:
        logger.warning("PID %d: Could not determine which GPU devices are in use", pid)

    logger.info("Locking CUDA process (PID: %d)...", pid)

    # Lock the CUDA process
    err, = cuda.cuCheckpointProcessLock(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        error_name, error_desc = get_cuda_error_details(err)
        if gpu_devices:
            log_gpu_memory_for_devices(gpu_devices, f"PID_{pid}_LOCK_FAILED")
        raise RuntimeError(
            f"Failed to lock CUDA process (PID {pid}, GPUs {gpu_devices}): "
            f"{error_name} - {error_desc}"
        )
    logger.info("CUDA process locked (PID: %d)", pid)

    # Checkpoint the CUDA process
    logger.info("Checkpointing CUDA process (PID: %d)...", pid)
    err, = cuda.cuCheckpointProcessCheckpoint(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        error_name, error_desc = get_cuda_error_details(err)
        if gpu_devices:
            log_gpu_memory_for_devices(gpu_devices, f"PID_{pid}_CHECKPOINT_FAILED")
        raise RuntimeError(
            f"Failed to checkpoint CUDA process (PID {pid}, GPUs {gpu_devices}): "
            f"{error_name} - {error_desc}"
        )
    
    logger.info("CUDA process checkpointed (PID: %d)", pid)
    
    # Log memory after successful checkpoint
    if gpu_devices:
        log_gpu_memory_for_devices(gpu_devices, f"PID_{pid}_AFTER")


def process_has_nvidia_fd(pid: int) -> bool:
    """Check if a process has any NVIDIA device file descriptors open.

    Args:
        pid: Process ID to check

    Returns:
        True if the process has /dev/nvidia* FDs open, False otherwise
    """
    fd_dir = f"/proc/{pid}/fd"
    if not os.path.exists(fd_dir):
        return False
    try:
        for fd in os.listdir(fd_dir):
            try:
                link = os.readlink(os.path.join(fd_dir, fd))
                if link.startswith("/dev/nvidia"):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def find_gpu_worker_pids(root_pid: int) -> list[int]:
    """Find all GPU worker processes in the process tree.

    Returns PIDs of leaf processes that use GPU (workers).
    """
    all_pids = collect_process_tree_pids(root_pid)

    # Find leaf processes (no children)
    leaf_pids = []
    for pid in all_pids:
        if process_is_leaf(pid):
            leaf_pids.append(pid)

    # Filter for GPU-using processes
    gpu_pids = []
    for pid in leaf_pids:
        if process_has_nvidia_fd(pid):
            gpu_pids.append(pid)

    return gpu_pids



def get_processes_with_nvidia_fds(pids: list[int]) -> list[int]:
    """Get list of PIDs that still have NVIDIA device file descriptors.

    Args:
        pids: List of PIDs to check

    Returns:
        List of PIDs that have /dev/nvidia* FDs open
    """
    return [pid for pid in pids if process_has_nvidia_fd(pid)]


def assert_no_nvidia_fds_in_tree(root_pid: int) -> None:
    """Assert that no processes in the tree have NVIDIA FDs after checkpoint.

    Args:
        root_pid: Root process ID of the tree to check

    Raises:
        RuntimeError: If any process still has NVIDIA FDs
    """
    all_pids = collect_process_tree_pids(root_pid)
    remaining = get_processes_with_nvidia_fds(list(all_pids))

    if remaining:
        # Collect detailed info about processes that still have FDs
        detailed_info = []
        for pid in remaining:
            try:
                comm = read_comm(pid)
                cmdline = read_cmdline(pid)
                detailed_info.append(f"PID {pid} ({comm}): {cmdline}")
            except Exception:
                detailed_info.append(f"PID {pid}")

        raise RuntimeError(
            f"CUDA checkpoint failed: {len(remaining)} process(es) still have "
            f"/dev/nvidia* FDs open.\n"
            f"Processes with NVIDIA FDs:\n" + "\n".join(detailed_info)
        )

    logger.info("Verified: No processes in tree have /dev/nvidia* FDs")


def format_cuda_checkpoint_results(succeeded: list[int],
                                  failed: list[tuple[int, str]]) -> str:
    """Format CUDA checkpoint results for logging.

    Args:
        succeeded: List of PIDs that were successfully checkpointed
        failed: List of (pid, error_message) tuples for failed checkpoints

    Returns:
        Formatted string summarizing the results
    """
    msg_parts = []

    if succeeded:
        msg_parts.append(f"CUDA checkpoint succeeded for {len(succeeded)} PID(s):")
        for pid in succeeded:
            devices = get_process_gpu_devices(pid)
            if devices:
                msg_parts.append(f"  PID {pid} (GPUs: {devices})")
            else:
                msg_parts.append(f"  PID {pid}")

    if failed:
        msg_parts.append(f"CUDA checkpoint failed for {len(failed)} PID(s):")
        for pid, error in failed:
            devices = get_process_gpu_devices(pid)
            if devices:
                msg_parts.append(f"  PID {pid} (GPUs: {devices}): {error}")
            else:
                msg_parts.append(f"  PID {pid}: {error}")

    return "\n".join(msg_parts)


def checkpoint_cuda_processes_from_pids(cuda_pids: list[int]) -> tuple[list[int], list[tuple[int, str]]]:
    """Checkpoint CUDA processes from a list of PIDs.

    This function attempts to checkpoint each CUDA process and returns
    success/failure results.

    Args:
        cuda_pids: List of PIDs to checkpoint

    Returns:
        Tuple of (succeeded, failed) where:
        - succeeded: List of PIDs that were successfully checkpointed
        - failed: List of (pid, error_message) tuples for failed checkpoints
    """
    if not cuda_pids:
        return [], []

    if not cuda_available:
        raise RuntimeError(
            "cuda-python is not installed; install 'cuda-python' to "
            "enable CUDA API checkpointing")

    succeeded: list[int] = []
    failed: list[tuple[int, str]] = []

    for pid in cuda_pids:
        try:
            checkpoint_cuda_process(pid)
            succeeded.append(pid)
        except Exception as e:
            error_msg = str(e)
            failed.append((pid, error_msg))
            logger.debug("cuCheckpoint failed for PID %d: %s", pid, error_msg)

    return succeeded, failed

