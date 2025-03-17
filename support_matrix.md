# Dynamo Support Matrix

This document provides the support matrix for Dynamo, including build,  hardware and software, like GPU architectures, and Python versions.

## Hardware Compatibility


| **CPU Architecture**  | **Supported** |
|-----------------------|---------------|
| **x86_64**            | Supported     |
| **ARM64**             | Experimental  |

> **Note**: While **x86_64** architecture is fully supported, **ARM64** support is experimental and may have limitations.

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
| **Linux**              | x86_64, aarch64    | Supported, Experimental |
| **macOS**              | x86_64, ARM64      | Supported*, Experimental|

> **Note**: 
> - **Linux**: **x86_64** architecture is supported.
> - **macOS**: Dynamo is supported on **x86_64** architecture, but the installation requires building from binaries on **macOS**. There is no pre-built wheel for macOS.

## Software Compatibility

| **Dependency**   | **Version** |
|------------------|-------------|
|**Base Container**|    25.01    |
| **vLLM**         |    0.7.2    |
|**TensorrtLLM**   |    TBD*     |
|**NIXL**          |    0.1.0    |
|**CompoundAI**    |    0.0.11   |

> **Note***: The specific version of TensorRT-LLM that will be supported by Dynamo is yet to be determined (TBD). 

## Build Support
**Dynamo** currently provides build support in the following ways:

- **Wheels**: Pre-built Python wheels are only available for **x86_64 Linux**. No wheels are available for other platforms at this time.
- **Container Images**: We distribute only the source code for container images, and only **x86_64 Linux** is supported for these. Users must build the container image from source if they require it.

## Wheel Installation Commands

Once you've confirmed your platform and architecture are compatible, you can install **Dynamo** using the Python wheel.

### Steps for Wheel Installation:

Python 3.10, 3.11, and 3.12 are supported Python Versions for Wheel Installation.

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
> **Note***: NIXL support is available only for Python 3.12.

### Steps for Container Build in local:

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
> **Note***: The default framework is Standard.


