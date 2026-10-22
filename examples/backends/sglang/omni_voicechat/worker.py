# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Adapt native VoiceChat to the frontend's typed bidirectional realtime engine."""

import asyncio
import json
import logging
import os
import uuid

from aiohttp import ClientSession, ClientTimeout, WSMsgType

MODEL = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
LOG = logging.getLogger("voicechat.worker")


def error(code, message):
    return {
        "type": "error",
        "event_id": f"event_{uuid.uuid4().hex}",
        "error": {"type": "invalid_request_error", "code": code, "message": message},
    }


class VoiceChatWorker:
    def __init__(self, session, url, instance_id):
        self.session, self.url, self.instance_id = session, url, str(instance_id)
        self.active = False

    async def generate(self, requests, context):
        # Reserve on the worker, so admission holds across frontend replicas.
        if self.active:
            yield error(
                "server_busy", "VoiceChat is in use; end the other conversation first"
            )
            return
        self.active = True
        stats = {"input_frames": 0, "input_acks": 0, "audio_chunks": 0}
        tasks = []
        try:
            async with self.session.ws_connect(
                self.url + "/v1/realtime", max_msg_size=1024 * 1024
            ) as ws:
                LOG.info(
                    "session_open worker=%s request=%s", self.instance_id, context.id()
                )

                async def send():
                    ended = False
                    async for event in requests:
                        event["event_id"] = (
                            event.get("event_id") or f"event_{uuid.uuid4().hex}"
                        )
                        kind = event["type"]
                        if kind == "session.update":
                            config = event["session"]
                            if config.get("output_modalities") not in (None, ["audio"]):
                                raise ValueError(
                                    "This demo supports output_modalities=['audio']"
                                )
                            if any(
                                config.get(key)
                                for key in ("instructions", "tools", "audio")
                            ):
                                raise ValueError(
                                    "This demo uses the checkpoint's fixed voice, prompt and PCM formats"
                                )
                            await ws.send_json(
                                {
                                    "type": kind,
                                    "event_id": event.get("event_id"),
                                    "session": {
                                        "type": "realtime",
                                        "model": MODEL,
                                        "output_modalities": ["audio"],
                                    },
                                }
                            )
                        elif kind == "input_audio_buffer.append" and not ended:
                            seq = stats["input_frames"]
                            await ws.send_json(
                                {
                                    **event,
                                    "sglang": {"seq": seq, "t_start_ms": seq * 80},
                                }
                            )
                            stats["input_frames"] += 1
                        elif kind == "input_audio_buffer.commit" and not ended:
                            # One continuous model response: commit drains this session's input.
                            ended = True
                            await ws.send_json(
                                {
                                    "type": "sglang.input_audio.end",
                                    "event_id": event.get("event_id"),
                                }
                            )
                        else:
                            raise ValueError(
                                f"Unsupported VoiceChat client event: {kind}"
                            )

                async def stop():
                    while not context.is_stopped():
                        await asyncio.sleep(0.05)
                    await ws.close()

                sender = asyncio.create_task(send())
                stopper = asyncio.create_task(stop())
                tasks = [sender, stopper]
                while True:
                    receiving = asyncio.create_task(ws.receive())
                    tasks.append(receiving)
                    done, _ = await asyncio.wait(
                        (receiving, sender, stopper),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if sender in done:
                        sender.result()
                        # Input EOF is distinct from cancellation; retain pending output.
                        done, _ = await asyncio.wait(
                            (receiving, stopper), return_when=asyncio.FIRST_COMPLETED
                        )
                    if stopper in done:
                        break
                    message = receiving.result()
                    tasks.remove(receiving)
                    if message.type == WSMsgType.ERROR:
                        raise ws.exception()
                    if message.type != WSMsgType.TEXT:
                        break
                    event = json.loads(message.data)
                    kind = event["type"]
                    if kind == "session.created":
                        session_id = event["session"]["id"]
                    elif kind == "session.updated":
                        granted = event["session"]["sglang"]["granted"]
                        if (
                            granted["input_audio_format"]["rate"],
                            granted["output_audio_format"]["rate"],
                        ) != (16000, 22050):
                            raise ValueError("Unexpected native PCM formats")
                        yield {
                            "type": kind,
                            "event_id": event["event_id"],
                            "session": {
                                "type": "realtime",
                                "id": session_id,
                                "model": MODEL,
                                "output_modalities": ["audio"],
                            },
                        }
                    elif kind == "sglang.input_audio.accepted":
                        stats["input_acks"] += 1
                    elif kind == "sglang.input_audio.drained":
                        yield {
                            "type": "input_audio_buffer.committed",
                            "event_id": event["event_id"],
                            "item_id": session_id + "-input",
                        }
                    elif kind.startswith("response.") or kind == "error":
                        event.pop("sglang", None)
                        if kind in ("response.created", "response.done"):
                            response = event["response"]
                            details = response.get("status_details") or {}
                            reason = details.get("reason")
                            # Native reasons differ from the typed OpenAI reason enum.
                            if details:
                                response["status_details"] = (
                                    None
                                    if response["status"] == "completed"
                                    else {"type": response["status"]}
                                )
                            event["response"].update(
                                {
                                    "max_output_tokens": "inf",
                                    "output_modalities": ["audio"],
                                }
                            )
                            event["response"]["metadata"] = {
                                "dynamo_worker_id": self.instance_id,
                                **{key: str(value) for key, value in stats.items()},
                                **({"native_finish_reason": reason} if reason else {}),
                            }
                        if kind == "response.output_audio.delta":
                            stats["audio_chunks"] += 1
                        yield event
        except Exception:
            LOG.exception("session_failed worker=%s", self.instance_id)
            yield error("backend_error", "VoiceChat session failed; see worker logs")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.active = False
            LOG.info(
                "session_close worker=%s stats=%s", self.instance_id, json.dumps(stats)
            )


async def main():
    from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
    from dynamo.runtime import DistributedRuntime

    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "etcd", "tcp", event_plane="zmq"
    )
    endpoint = runtime.endpoint("voicechat.voicechat.realtime")
    url = os.environ.get("OMNI_URL", "http://127.0.0.1:8097")
    try:
        async with ClientSession(
            timeout=ClientTimeout(total=None, sock_connect=5)
        ) as session:
            while True:
                try:
                    async with session.get(url + "/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except (OSError, TimeoutError):
                    pass
                await asyncio.sleep(2)
            worker = VoiceChatWorker(session, url, endpoint.connection_id())
            await register_model(
                ModelInput.Text,
                ModelType.Realtime,
                endpoint,
                MODEL,
                model_name=MODEL,
                worker_type=WorkerType.Aggregated,
            )
            await endpoint.serve_bidirectional_endpoint(worker.generate)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
