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
| **Linux**              | x86_64             | Supported               |
| **Linux**              | ARM64              | Experimental            |

> **Note**: For **Linux**, the **ARM64** support is experimental and may have limitations.

## Software Compatibility

| **Dependency**   | **Version** |
|------------------|-------------|
|**Base Container**|    25.01    |
| **vLLM**         |    0.7.2    |
|**TensorRT-LLM**  |    0.19.0*  |
|**NIXL**          |    0.1.0    |

> **Note**:
> *The specific version of TensorRT-LLM (planned v0.19.0) that will be supported by Dynamo is subject to change.

## Build Support
**Dynamo** currently provides build support in the following ways:

- **Wheels**: Pre-built Python wheels are only available for **x86_64 Linux**. No wheels are available for other platforms at this time.
- **Container Images**: We distribute only the source code for container images, and only **x86_64 Linux** is supported for these. Users must build the container image from source if they require it.

Once you've confirmed that your platform and architecture are compatible, you can install **Dynamo** by following the instructions in the [Quick Start Guide](https://github.com/ai-dynamo/dynamo/?tab=readme-ov-file#quick-start).
