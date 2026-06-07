import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

_EVENTS_PATH = "/events"
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)
_QUEUE_SIZE = 1000


class FlexPriceClient:
    """Async client that emits LLM usage events to FlexPrice in the background.

    Enqueue is non-blocking — the caller returns immediately and the background
    worker drains the queue independently, so billing never adds latency to the
    request path.
    """

    def __init__(self, api_host: str, api_key: str) -> None:
        self._events_url = f"https://{api_host}{_EVENTS_PATH}"
        self._headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue(
            maxsize=_QUEUE_SIZE
        )
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            headers=self._headers,
            connector=aiohttp.TCPConnector(ssl=True),
        )
        self._worker_task = asyncio.create_task(
            self._worker(), name="flexprice-event-worker"
        )

    async def stop(self) -> None:
        await self._queue.put(None)  # sentinel — drain then exit
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()
        if self._session:
            await self._session.close()

    def enqueue(
        self,
        event_name: str,
        external_customer_id: str,
        properties: Dict[str, Any],
        source: str = "",
        event_id: Optional[str] = None,
    ) -> None:
        """Non-blocking enqueue. Drops silently when the queue is full."""
        event: Dict[str, Any] = {
            "event_name": event_name,
            "external_customer_id": external_customer_id,
            "properties": {k: str(v) for k, v in properties.items()},
            "source": source,
            "event_id": event_id or str(uuid.uuid4()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "FlexPrice event queue full; dropping event for customer=%s",
                external_customer_id,
            )

    async def _worker(self) -> None:
        while True:
            event = await self._queue.get()
            if event is None:
                while not self._queue.empty():
                    item = self._queue.get_nowait()
                    if item is not None:
                        await self._send(item)
                break
            await self._send(event)

    async def _send(self, event: Dict[str, Any]) -> None:
        if not self._session:
            return
        payload = {
            "event_name": event["event_name"],
            "external_customer_id": event["external_customer_id"],
            "properties": event["properties"],
            "source": event.get("source", ""),
            "event_id": event.get("event_id", ""),
            "timestamp": event.get("timestamp", ""),
        }
        try:
            async with self._session.post(
                self._events_url, json=payload, timeout=_REQUEST_TIMEOUT
            ) as resp:
                if resp.status < 200 or resp.status >= 300:
                    logger.warning(
                        "FlexPrice API returned %d for event %s",
                        resp.status, event.get("event_name"),
                    )
        except Exception as exc:
            logger.warning("Failed to emit FlexPrice event: %s", exc)
