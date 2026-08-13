#pragma once

#include "v1/pagebroker.pb.h"

namespace snapshot::pagebroker {
using v1::Failure;
using v1::IOEngine;
using v1::PrepareStagedCheckpointRequest;
using v1::Request;
using v1::Response;
using v1::StagedRestoreRequest;
using v1::StorageBackend;
}  // namespace snapshot::pagebroker
