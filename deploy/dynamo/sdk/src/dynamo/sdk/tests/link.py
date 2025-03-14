# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# linking syntax example


from dynamo.sdk.tests.pipeline import Backend, Frontend, Middle

# print("INITIAL DEPENDENCIES")
# print("Frontend dependencies", Frontend.dependencies)
# print("Middle dependencies", Middle.dependencies)
# print("Backend dependencies", Backend.dependencies)

# print("\n\n\n")

print()
Frontend.link(Middle).build()

"""
---------------
class FrontEnd:
    decode_worker = depends(DecodeWorker)
    preproc = depends(Preproc)              # optional

class Processor:
    kv_router = depends(KvRouter)

class KvRouter:
    decode_worker = depends(DecodeWorker)

class DecodeWorker:
    prefill_worker = depends(PrefillWorker) # optional
---------------

# Optional
Frontend -> Preproc

if cli.enable_prefill:
    Frontend -> Preproc -> KvRouter

if cli.enable_prefill:
    DecodeWorker -> PrefillWorker
-----------

Monolith
Frontend.link(DecodeWorker).build()

Kv aware monolith
Frontend.link(Processor).link(DecodeWorker).build()
Frontend.link(KvRouter)

Kv off + Disag on
Frontend.link(DecodeWorker).link(PrefillWorker).build()
if kv_disabled:
    Processor.unlink(KvRouter)

Kv on + Disag On
Frontend.link(Processor)

"""

print("Frontend dependencies", Frontend.dependencies)
print("Middle dependencies", Middle.dependencies)
print("Backend dependencies", Backend.dependencies)
