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

| **GPU Architecture**                | **Supported** |
|-------------------------------------|---------------|
| **NVIDIA Blackwell Architecture**   | Supported     |
| **NVIDIA Grace Hopper Superchip**   | Supported     |
| **NVIDIA Hopper Architecture**      | Supported     |
| **NVIDIA Ada Lovelace Architecture**| Supported     |
| **NVIDIA Ampere Architecture**      | Supported     |

## Platform Architecture Compatibility

**Dynamo** is compatible with the following platforms:

| **Operating System**   | **Architecture**   | **Supported** |
|------------------------|--------------------|---------------|
| **Linux**              | x86_64, aarch64    | Supported     |
| **macOS**              | x86_64, ARM64      | Supported*  |

> **Note**: 
> - **macOS**: Dynamo is supported on both **x86_64** and **ARM64** architectures, but the installation requires building from binaries on **macOS**. There is no pre-built wheel for macOS.
> - **Linux**: **x86_64** and **aarch64** architectures are fully supported.


## Software Compatibility

| **Dependency**   | **Version** |
|------------------|-------------|
|**Base Container**|    25.01    |
| **vLLM**         |    0.7.2    |
|**TensorrtLLM**   |    0.18.0dev|
|**NIXL**          |    0.1.0    |
|**CompoundAI**    |    0.0.11   |


## Build Support
**Dynamo** currently provides build support in the following ways:

- **Wheels**: Pre-built Python wheels are only available for **x86_64 Linux**. No wheels are available for other platforms at this time.
- **Container Images**: We distribute only the source code for container images, and only **x86_64 Linux** is supported for these. Users must build the container image from source if they require it.

## Wheel Installation Commands

Once you've confirmed your platform and architecture are compatible, you can install **Dynamo** using the Python wheel.

### Steps for Wheel Installation:

pip install ai-dynamo==0.1.0