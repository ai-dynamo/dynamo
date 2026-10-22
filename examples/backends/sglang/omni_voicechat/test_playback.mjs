// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Run the real AudioWorklet algorithm at VoiceChat's rate with a catch-up backlog.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const location = process.argv[2];
assert(location, "Supply the worklet.js path or deployed URL");
let source;
if (location.startsWith("http")) {
  const response = await fetch(location);
  assert(response.ok, `HTTP ${response.status}`);
  source = await response.text();
} else {
  source = await readFile(location, "utf8");
}
const processors = {};
vm.runInNewContext(source, {
  sampleRate: 22050,
  AudioWorkletProcessor: class {
    constructor() { this.port = { postMessage() {} }; }
  },
  registerProcessor: (name, processor) => { processors[name] = processor; },
});
const player = new processors["playback-processor"]({
  processorOptions: { initialSamples: 1, maxSamples: 100000, keepSamples: 100000 },
});
player.port.onmessage({ data: {
  type: "push", samples: new Float32Array(22050).fill(0.1), responseId: "reply",
} });
let invalid = 0;
let audible = 0;
for (let frame = 0; frame < 80; frame += 1) {
  const output = new Float32Array(128);
  player.process([], [[output]]);
  invalid += output.filter((sample) => !Number.isFinite(sample)).length;
  audible += output.filter((sample) => Math.abs(sample) > 0.001).length;
}
assert(player.rate > 1, "Catch-up path was not exercised");
assert.equal(invalid, 0, "22.05 kHz overlap-add produced non-finite speaker samples");
assert(audible > 0, "Playback became silent");
console.log(JSON.stringify({ sampleRate: 22050, samples: 10240, invalid, audible, catchUpRate: player.rate }));
