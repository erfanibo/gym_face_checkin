"""
Tracks connected reception-panel WebSocket clients and broadcasts JSON events
to all of them (new face in queue, someone checked in, a queue item was
resolved by an operator, ...).

The camera loop runs in a plain background thread (not a coroutine), so it
cannot call `await manager.broadcast(...)` directly. Instead, FaceEngine
schedules the broadcast onto the main asyncio loop with
`asyncio.run_coroutine_threadsafe` (see face_engine.py).
"""
import asyncio
import json
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._connections.discard(ws)

    async def broadcast(self, message: dict[str, Any]):
        payload = json.dumps(message, ensure_ascii=False)
        async with self._lock:
            targets = list(self._connections)

        dead = []
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)


manager = ConnectionManager()
