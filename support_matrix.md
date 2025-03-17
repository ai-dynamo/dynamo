# Dynamo Support Matrix

This document provides the support matrix for Dynamo, including hardware, software and build instructions.

## Hardware Compatibility


| **CPU Architecture**  | **Status**    |
|-----------------------|---------------|
| **x86_64**            | Supported     |
| **ARM64**             | Experimental  |

> **Note**: While **x86_64** architecture is supported on systems with a minimum of 32 GB RAM and at least 4 CPU cores. The **ARM64** support is experimental and may have limitations.

### GPU Compatibility

If you are using a **GPU**, the following GPU models and architectures are supported:

| **GPU Architecture**                | **Status**    |
|-------------------------------------|---------------|
| **NVIDIA Blackwell Architecture**   | Supported     |
| **NVIDIA Hopper Architecture**      | Supported     |
| **NVIDIA Ada Lovelace Architecture**| Supported     |
| **NVIDIA Ampere Architecture**      | Supported     |

## Platform Architecture Compatibility

**Dynamo** is compatible with the following platforms:

| **Operating System**   | **Architecture**   | **Status**              |
|------------------------|--------------------|-------------------------|
| **Linux**              | x86_64, ARM64      | Supported, Experimental |

> **Note**:
> - **Linux**: The **ARM64** support is experimental and may have limitations.

## Software Compatibility

| **Dependency**   | **Version** |
|------------------|-------------|
|**Base Container**|    25.01    |
| **vLLM**         |    0.7.2    |
|**TensorRT-LLM**  |    0.19.0*  |
|**NIXL**          |    0.1.0    |

> **Note**: *The specific version of TensorRT-LLM (planned v0.19.0) that will be supported by Dynamo is subject to change.

## Build Support
**Dynamo** currently provides build support in the following ways:

- **Wheels**: Pre-built Python wheels are only available for **x86_64 Linux**. No wheels are available for other platforms at this time.
- **Container Images**: We distribute only the source code for container images, and only **x86_64 Linux** is supported for these. Users must build the container image from source if they require it.

## Wheel Installation Commands

Once you've confirmed your platform and architecture are compatible, you can install **Dynamo** using the Python wheel.

### Steps for Wheel Installation:

Python 3.10, 3.11, and 3.12 are supported Python Versions for Wheel Installation.
> **Note**: The recommended version is Python 3.12.

1. **Install the base version of ai-dynamo**:
```bash
pip install ai-dynamo==0.1.0
```
2. **For the patched version of vLLM**:
```bash
pip install ai-dynamo vllm==0.7.2+dynamo
```
3. **For NIXL support, To install ai-dynamo with NIXL support, run the following:**:
```bash
pip install ai-dynamo nixl vllm==0.7.2+dynamo
```
> **Note**: NIXL support is available only for Python 3.12.

### Steps for Container Build in Local:

First, clone the `dynamo` repository to your local machine:
   ```bash
   git clone https://github.com/ai-dynamo/dynamo.git
   cd dynamo
   ./container/build.sh
   ```

   For building dockerfile for vllm framework run,
   ```bash
   ./container/build.sh --framework vllm
   ```
> **Note**: The default framework is vllm.


