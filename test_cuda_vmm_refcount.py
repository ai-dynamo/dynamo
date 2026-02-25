"""
Automated CUDA VMM reference counting test.

Measures nvidia-smi at each milestone to determine whether cuMemRelease
on the creator frees physical memory while another process holds an imported handle.
"""

import os
import socket
import struct
import array
import multiprocessing
import subprocess
import time
import sys


ALLOC_SIZE = 512 * 1024 * 1024  # 512 MiB for clear signal


def smi_free_mib(device=0):
    """nvidia-smi reported free memory in MiB (ground truth, external to CUDA runtime)."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            f"-i={device}",
        ],
        text=True,
    )
    return int(out.strip())


def smi_used_mib(device=0):
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            f"-i={device}",
        ],
        text=True,
    )
    return int(out.strip())


def send_fd(sock, fd):
    fds = array.array("i", [fd])
    sock.sendmsg([b"\x00"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])


def recv_fd(sock):
    msg, ancdata, flags, addr = sock.recvmsg(
        1, socket.CMSG_SPACE(struct.calcsize("i"))
    )
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            return struct.unpack("i", cmsg_data[:4])[0]
    raise RuntimeError("No FD received")


def make_prop():
    from cuda.bindings import driver as cuda
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = 0
    prop.requestedHandleTypes = (
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    return prop


def get_aligned_size(prop):
    from cuda.bindings import driver as cuda
    err, gran = cuda.cuMemGetAllocationGranularity(
        prop,
        cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
    )
    assert err == cuda.CUresult.CUDA_SUCCESS
    return ((ALLOC_SIZE + gran - 1) // gran) * gran


def init_cuda():
    from cuda.bindings import driver as cuda
    cuda.cuInit(0)
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    err, ctx = cuda.cuDevicePrimaryCtxRetain(device)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuCtxSetCurrent(ctx)
    assert err == cuda.CUresult.CUDA_SUCCESS
    return cuda


# ── Server process ────────────────────────────────────────────────────────────

def server_proc(sock_path, ev_go_create, ev_created, ev_go_release, ev_released, ev_shutdown):
    cuda = init_cuda()
    prop = make_prop()
    aligned = get_aligned_size(prop)

    ev_go_create.wait()

    err, handle = cuda.cuMemCreate(aligned, prop, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS, f"cuMemCreate: {err}"

    err, fd = cuda.cuMemExportToShareableHandle(
        handle,
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        0,
    )
    assert err == cuda.CUresult.CUDA_SUCCESS, f"export: {err}"

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    ev_created.set()  # tell client to connect

    conn, _ = srv.accept()
    send_fd(conn, fd)
    os.close(fd)

    ev_go_release.wait()

    (err,) = cuda.cuMemRelease(handle)
    assert err == cuda.CUresult.CUDA_SUCCESS, f"cuMemRelease: {err}"

    ev_released.set()

    ev_shutdown.wait()
    conn.close()
    srv.close()


# ── Client process ────────────────────────────────────────────────────────────

def client_proc(sock_path, ev_created, ev_imported, ev_go_client_release, ev_client_released, ev_shutdown):
    cuda = init_cuda()
    prop = make_prop()
    aligned = get_aligned_size(prop)

    ev_created.wait()
    time.sleep(0.3)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    fd = recv_fd(sock)

    err, imported_handle = cuda.cuMemImportFromShareableHandle(
        fd,
        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    )
    assert err == cuda.CUresult.CUDA_SUCCESS, f"import: {err}"
    os.close(fd)

    err, va = cuda.cuMemAddressReserve(aligned, 0, 0, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuMemMap(va, aligned, 0, imported_handle, 0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    access = cuda.CUmemAccessDesc()
    access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = 0
    access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    (err,) = cuda.cuMemSetAccess(va, aligned, [access], 1)
    assert err == cuda.CUresult.CUDA_SUCCESS

    ev_imported.set()

    ev_go_client_release.wait()

    (err,) = cuda.cuMemUnmap(va, aligned)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuMemRelease(imported_handle)
    assert err == cuda.CUresult.CUDA_SUCCESS
    (err,) = cuda.cuMemAddressFree(va, aligned)
    assert err == cuda.CUresult.CUDA_SUCCESS

    ev_client_released.set()

    ev_shutdown.wait()
    sock.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    sock_path = "/tmp/test_cuda_vmm_refcount.sock"
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    ev_go_create = multiprocessing.Event()
    ev_created = multiprocessing.Event()
    ev_imported = multiprocessing.Event()
    ev_go_release = multiprocessing.Event()
    ev_released = multiprocessing.Event()
    ev_go_client_release = multiprocessing.Event()
    ev_client_released = multiprocessing.Event()
    ev_shutdown_server = multiprocessing.Event()
    ev_shutdown_client = multiprocessing.Event()

    readings = {}

    SETTLE = 2  # seconds to let nvidia-smi settle

    print("=" * 65)
    print("CUDA VMM Reference Counting Test  (automated, nvidia-smi based)")
    print(f"Allocation size: {ALLOC_SIZE // (1<<20)} MiB")
    print("=" * 65)
    print()

    # ── Step 0: Baseline ──
    time.sleep(SETTLE)
    readings["0_baseline"] = smi_used_mib()
    print(f"[Step 0] Baseline              used: {readings['0_baseline']} MiB")

    p_server = multiprocessing.Process(
        target=server_proc,
        args=(sock_path, ev_go_create, ev_created, ev_go_release, ev_released, ev_shutdown_server),
    )
    p_client = multiprocessing.Process(
        target=client_proc,
        args=(sock_path, ev_created, ev_imported, ev_go_client_release, ev_client_released, ev_shutdown_client),
    )
    p_server.start()
    p_client.start()

    # ── Step 1: Server creates + client imports ──
    ev_go_create.set()
    ev_imported.wait()
    time.sleep(SETTLE)
    readings["1_after_alloc_import"] = smi_used_mib()
    delta1 = readings["1_after_alloc_import"] - readings["0_baseline"]
    print(f"[Step 1] After alloc+import    used: {readings['1_after_alloc_import']} MiB  (Δ from baseline: +{delta1} MiB)")

    # ── Step 2: Server releases handle (client still holds) ──
    ev_go_release.set()
    ev_released.wait()
    time.sleep(SETTLE)
    readings["2_after_server_release"] = smi_used_mib()
    delta2 = readings["2_after_server_release"] - readings["1_after_alloc_import"]
    print(f"[Step 2] After server release  used: {readings['2_after_server_release']} MiB  (Δ from step 1: {delta2:+d} MiB)  ← KEY MEASUREMENT")

    # ── Step 3: Client releases handle (all refs gone, processes alive) ──
    ev_go_client_release.set()
    ev_client_released.wait()
    time.sleep(SETTLE)
    readings["3_after_client_release"] = smi_used_mib()
    delta3 = readings["3_after_client_release"] - readings["2_after_server_release"]
    print(f"[Step 3] After client release  used: {readings['3_after_client_release']} MiB  (Δ from step 2: {delta3:+d} MiB)")

    # ── Step 4: Kill both processes ──
    ev_shutdown_server.set()
    ev_shutdown_client.set()
    p_server.join(timeout=5)
    p_client.join(timeout=5)
    if p_server.is_alive():
        p_server.kill()
        p_server.join()
    if p_client.is_alive():
        p_client.kill()
        p_client.join()
    time.sleep(SETTLE)
    readings["4_after_process_death"] = smi_used_mib()
    delta4 = readings["4_after_process_death"] - readings["0_baseline"]
    print(f"[Step 4] After process death   used: {readings['4_after_process_death']} MiB  (Δ from baseline: {delta4:+d} MiB)")

    if os.path.exists(sock_path):
        os.unlink(sock_path)

    # ── Analysis ──
    print()
    print("=" * 65)
    print("ANALYSIS")
    print("=" * 65)
    print()

    alloc_mib = ALLOC_SIZE // (1 << 20)
    threshold = alloc_mib * 0.7

    print(f"Expected allocation: {alloc_mib} MiB")
    print()

    # Check step 1: did allocation show up?
    if delta1 >= threshold:
        print(f"  Step 1: ✓ Allocation visible (+{delta1} MiB). nvidia-smi tracks VMM allocs.")
    else:
        print(f"  Step 1: ✗ Allocation NOT visible (+{delta1} MiB). nvidia-smi may not track VMM allocs!")
        print(f"          Cannot draw conclusions. Test is inconclusive.")
        sys.exit(1)

    # Check step 2: did server release free the memory?
    if delta2 <= -threshold:
        print(f"  Step 2: Memory WAS freed by server cuMemRelease ({delta2:+d} MiB).")
        print(f"          → Client imported handle does NOT keep memory alive.")
        print(f"          → Force eviction via GMS cuMemRelease WORKS. No OOM gap.")
    elif abs(delta2) < 10:
        print(f"  Step 2: Memory was NOT freed by server cuMemRelease ({delta2:+d} MiB).")
        print(f"          → Client imported handle KEEPS memory alive (ref counting).")
        print(f"          → Force eviction via GMS cuMemRelease is INSUFFICIENT.")
    else:
        print(f"  Step 2: Partial change ({delta2:+d} MiB). Inconclusive.")

    # Check step 3: did client release free it?
    if delta3 <= -threshold:
        print(f"  Step 3: Memory freed after client release ({delta3:+d} MiB).")
        print(f"          → Confirms ref counting: last handle release frees memory.")
    elif abs(delta3) < 10:
        print(f"  Step 3: No change after client release ({delta3:+d} MiB).")
        if delta2 <= -threshold:
            print(f"          → Expected: memory was already freed at step 2.")
        else:
            print(f"          → Memory held by CUDA context or driver cache, not just handles.")
    else:
        print(f"  Step 3: Partial change ({delta3:+d} MiB). Inconclusive.")

    # Check step 4: back to baseline?
    if abs(delta4) < 20:
        print(f"  Step 4: ✓ Back to baseline ({delta4:+d} MiB). No leak.")
    else:
        print(f"  Step 4: ✗ NOT back to baseline ({delta4:+d} MiB). Possible driver-level caching.")

    print()
    print("=" * 65)
    print(f"Server exit: {p_server.exitcode}, Client exit: {p_client.exitcode}")
