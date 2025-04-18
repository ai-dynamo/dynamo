# Dynamo Support Matrix

This document provides the support matrix for Dynamo, including hardware, software and build instructions.

## Hardware Compatibility


| **CPU Architecture**  | **Status**    |
|-----------------------|---------------|
| **x86_64**            | Supported     |
| **ARM64**             | Experimental  |

> **Note**: While **x86_64** architecture is supported on systems with a minimum of 32 GB RAM and at least 4 CPU cores, the **ARM64** support is experimental and may have limitations.

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

| **Operating System** | **Version** | **Architecture** | **Status**   |
|----------------------|-------------|------------------|--------------|
| **Ubuntu**           | 22.04       | x86_64           | Supported    |
| **Ubuntu**           | 24.04       | x86_64           | Supported    |
| **Ubuntu**           | 24.04       | ARM64            | Experimental |
| **CentOS Stream**    | 9           | x86_64           | Experimental |

> **Note**: For **Linux**, the **ARM64** support is experimental and may have limitations. Wheels are built using a manylinux_2_28-compatible environment and they have been validated on CentOS 9 and Ubuntu (22.04, 24.04). Compatibility with other Linux distributions is expected but has not been officially verified yet.

## Software Compatibility
### Runtime Dependency
| **Python Package** | **Version**   | glibc version        |
|--------------------|---------------|----------------------|
| ai-dynamo          |    0.1.1      |     >=2.28           |
| ai-dynamo-runtime  |    0.1.1      |     >=2.28           |
| ai-dynamo-vllm     |  0.7.2.post1* | >=2.28 (recommended) |
| NIXL               |    0.1.1      |     >=2.27           |

### Build Dependency
| **Build Dependency** | **Version** |
|----------------------|-------------|
| **Base Container**   |    [25.01](https://hub.docker.com/layers/nvidia/cuda/12.8.1-runtime-ubuntu24.04/images/sha256-828c4d878adcaa4265d80c95d8ec877149b49bb2419a4cf3bb6aa889bbb7ca2e)    |
| **ai-dynamo-vllm**   |0.7.2.post1* |
| **TensorRT-LLM**     |    0.19.0** |
| **NIXL**             |    0.1.1    |

> **Note**:
> - *ai-dynamo-vllm v0.7.2.post1* is a customized patch of v0.7.2 from vLLM.
> - **The specific version of TensorRT-LLM (planned v0.19.0) that will be supported by Dynamo is subject to change.


## Build Support
**Dynamo** currently provides build support in the following ways:

- **Wheels**: Pre-built Python wheels are only available for **x86_64 Linux**. No wheels are available for other platforms at this time.
- **Container Images**: We distribute only the source code for container images, and only **x86_64 Linux** is supported for these. Users must build the container image from source if they require it.

Once you've confirmed that your platform and architecture are compatible, you can install **Dynamo** by following the instructions in the [Quick Start Guide](https://github.com/ai-dynamo/dynamo/blob/main/README.md#installation).
