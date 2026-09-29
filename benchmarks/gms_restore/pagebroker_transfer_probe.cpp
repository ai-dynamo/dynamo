// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// Benchmark adapter: calls the unchanged qualified PageBroker transfer implementation.
#include "cuda_posix_transfer.hpp"
#include <cstdio>
#include <exception>
#include <string>
using namespace snapshot::pagebroker::cuda;
struct Probe { TransferBuffers buffers; CUstream stream = nullptr;
  Probe(size_t slots, size_t chunk) : buffers(TransferOptions{slots, chunk, true}) {} };
extern "C" void* probe_create(unsigned slots, size_t chunk) {
  try {
    auto* p = new Probe(slots, chunk); CUcontext ctx = nullptr; std::string error;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx ||
        cuStreamCreate(&p->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
        !p->buffers.Initialize(ctx, &error)) {
      std::fprintf(stderr, "probe setup: %s\n", error.c_str()); delete p; return nullptr;
    }
    return p;
  } catch(const std::exception& e) {std::fprintf(stderr,"probe setup: %s\n",e.what());return nullptr;}
}
extern "C" int probe_transfer(void* ptr, int fd, unsigned long long va, size_t size, double* times) {
  try {
    auto* p = static_cast<Probe*>(ptr); TransferMetrics m; std::string error;
    if (!p->buffers.Transfer(fd, va, size, p->stream, TransferOperation::kRestore, &m, &error)) {
      std::fprintf(stderr,"probe transfer: %s\n",error.c_str());return -1;
    }
    times[0]=m.total_seconds;times[1]=m.setup_seconds;times[2]=m.pipeline_seconds;
    times[3]=m.storage_io_seconds;times[4]=m.cuda_wait_seconds;
    return 0;
  } catch(const std::exception& e) {std::fprintf(stderr,"probe transfer: %s\n",e.what());return -1;}
}
extern "C" void probe_destroy(void* ptr) {
  auto* p=static_cast<Probe*>(ptr); cuStreamSynchronize(p->stream);cuStreamDestroy(p->stream);delete p;
}
