# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
GPU Fault Injector Agent - Runs as DaemonSet on GPU nodes.

This agent provides privileged access for XID error injection:
- XID injection via nsenter+kmsg (writes to host's /dev/kmsg)
- Triggers NVSentinel syslog-health-monitor detection
- Initiates complete fault tolerance workflow

Accepts ANY XID error code for testing flexibility.
Pre-defined messages for all DCGM/NVSentinel monitored XIDs:
- Devastating: 79, 74, 48, 94, 95, 119, 120, 140
- Memory: 31, 32, 43, 63, 64
- PCIe: 38, 39, 42
- Thermal: 60, 61, 62
- Power: 54, 56, 57
- Graphics: 13, 45, 69

Unknown XIDs use generic error message format.
NVSentinel detects XIDs and handles actions based on its own rules.
See gpu_xid_injector.py for complete XID descriptions.
"""

import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Optional, Type

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import kernel-level XID injector (for XID 79 via nsenter+kmsg)
GPUXIDInjectorKernel: Optional[Type[Any]] = None
try:
    from gpu_xid_injector import (  # type: ignore[no-redef, assignment]
        GPUXIDInjectorKernel,
    )

    KERNEL_XID_AVAILABLE = True
except ImportError:
    logger.warning("Kernel-level XID injector not available")
    KERNEL_XID_AVAILABLE = False


# ============================================================================
# Models and Enums
# ============================================================================


class XIDInjectRequest(BaseModel):
    """Request model for XID error injection via nsenter+kmsg"""

    fault_id: str
    xid_type: int
    gpu_id: int = 0
    duration: Optional[int] = None


# ============================================================================
# GPU Fault Injector
# ============================================================================


class GPUFaultInjector:
    """
    GPU fault injection operations with DCGM integration.

    Supports ANY XID injection via nsenter+kmsg (27+ pre-defined messages).
    Accepts any XID value (1-1000) for comprehensive fault tolerance testing.
    """

    def __init__(self):
        self.active_faults: dict[str, dict[str, Any]] = {}
        self.node_name = os.getenv("NODE_NAME", "unknown")
        self.dcgm_available = self._check_dcgm()
        self.gpu_count = self._get_gpu_count()

        # Initialize kernel-level XID injector (XID 79 via nsenter+kmsg)
        self.kernel_xid_injector = None
        self.kernel_xid_available = False
        if KERNEL_XID_AVAILABLE and GPUXIDInjectorKernel is not None:
            try:
                self.kernel_xid_injector = GPUXIDInjectorKernel()
                self.kernel_xid_available = self.kernel_xid_injector.privileged
                logger.info(
                    f"Kernel-level XID injector initialized (privileged: {self.kernel_xid_available})"
                )
            except Exception as e:
                logger.warning(f"Kernel XID injector not available: {e}")

        logger.info(f"GPU Fault Injector initialized on node: {self.node_name}")
        logger.info(f"DCGM available: {self.dcgm_available}")
        logger.info(f"GPU count: {self.gpu_count}")
        logger.info(f"XID 79 injection (nsenter+kmsg): {self.kernel_xid_available}")

    def _check_dcgm(self) -> bool:
        """Check if DCGM is available"""
        try:
            result = subprocess.run(
                ["dcgmi", "discovery", "-l"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"DCGM not available: {e}")
            return False

    def _get_gpu_count(self) -> int:
        """Get number of GPUs on this node"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip().split("\n")[0])
            return 0
        except Exception as e:
            logger.error(f"Failed to get GPU count: {e}")
            return 0

    def _run_command(self, command: list[str], timeout: int = 30) -> tuple[bool, str]:
        """Run shell command with timeout"""
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="GPU Fault Injector Agent", version="1.0.0")
injector = GPUFaultInjector()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "node": injector.node_name,
        "gpu_count": injector.gpu_count,
        "dcgm_available": injector.dcgm_available,
        "active_faults": len(injector.active_faults),
    }


@app.post("/inject-xid")
async def inject_xid(request: XIDInjectRequest):
    """
    Inject ANY XID error via nsenter+kmsg (triggers NVSentinel detection).

    Accepts any XID error code (1-1000) for maximum testing flexibility.

    Pre-defined messages for all DCGM/NVSentinel monitored XIDs:

    Devastating (always FAIL):
    - 79: GPU fell off bus | 74: NVLink error | 48: ECC DBE | 94/95: ECC errors
    - 119/120: GSP errors | 140: ECC unrecovered

    Subsystem (may WARN/escalate):
    - Memory: 31, 32, 43, 63, 64 (MMU, PBDMA, page retirement)
    - PCIe: 38, 39, 42 (bus, fabric, replay rate)
    - Thermal: 60, 61, 62 (temperature limits)
    - Power: 54, 56, 57 (power/clock state)
    - Graphics: 13, 45, 69 (SM exceptions)

    Unknown XIDs use generic error message - NVSentinel will parse and handle
    based on its own XID database.
    """
    logger.info(
        f"Received XID {request.xid_type} injection request for GPU {request.gpu_id}"
    )

    # Validate XID type is a reasonable integer (basic sanity check)
    if (
        not isinstance(request.xid_type, int)
        or request.xid_type < 1
        or request.xid_type > 1000
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid XID type: {request.xid_type}. "
                f"XID must be an integer between 1-1000. "
                f"Common XIDs: 79 (bus error), 74 (NVLink), 48/94/95 (ECC errors)."
            ),
        )

    if not injector.kernel_xid_available or not injector.kernel_xid_injector:
        raise HTTPException(
            status_code=503,
            detail=f"Kernel-level XID injector not available. XID {request.xid_type} requires privileged access to syslog/kmsg.",
        )

    # Use the generic inject_xid method which supports multiple XID types
    success, message = injector.kernel_xid_injector.inject_xid(
        xid_type=request.xid_type, gpu_id=request.gpu_id
    )

    if not success:
        raise HTTPException(status_code=500, detail=message)

    # Track the fault
    injector.active_faults[request.fault_id] = {
        "type": f"xid_{request.xid_type}",
        "gpu_id": request.gpu_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "status": "injected",
        "node": injector.node_name,
        "fault_id": request.fault_id,
        "xid_type": request.xid_type,
        "gpu_id": request.gpu_id,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/faults")
async def list_active_faults():
    """List active faults on this node"""
    return {
        "node": injector.node_name,
        "active_faults": list(injector.active_faults.keys()),
        "count": len(injector.active_faults),
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8083,
        log_level="info",
    )
