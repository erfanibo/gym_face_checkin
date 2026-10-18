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


MAX_CONNECTIONS_PER_MEMBER = 5

# Which manager-side events matter to a member's own panel, and which part of
# it must be reloaded. Members only ever receive a bare "refresh <scope>"
# signal for THEIR OWN id -- never the event payload itself -- and then
# re-fetch through their normal, authenticated /api/me/* endpoints.
_MEMBER_SCOPE_BY_EVENT = {
    "checkin": ("user_id", "attendance"),
    "membership_updated": ("id", "membership"),
    "member_updated": ("id", "profile"),
    "member_deleted": ("id", "profile"),
    "member_password_changed": ("id", "profile"),
    "workout_updated": ("user_id", "workout"),
}
_MEMBER_SCOPE_FOR_ALL = {"club_updated": "all", "backup_restored": "all"}


class ConnectionManager:
    def __init__(self):
        self._connections: set[WebSocket] = set()
        self._member_connections: dict[int, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._connections.discard(ws)

    async def connect_member(self, ws: WebSocket, user_id: int) -> bool:
        """Registers a member panel socket. Returns False (socket not accepted)
        when this member already has too many open connections."""
        async with self._lock:
            if len(self._member_connections.get(user_id, ())) >= MAX_CONNECTIONS_PER_MEMBER:
                return False
        await ws.accept()
        async with self._lock:
            self._member_connections.setdefault(user_id, set()).add(ws)
        return True

    async def disconnect_member(self, ws: WebSocket, user_id: int):
        async with self._lock:
            group = self._member_connections.get(user_id)
            if group is not None:
                group.discard(ws)
                if not group:
                    del self._member_connections[user_id]

    async def _notify_members(self, message: dict[str, Any]):
        event = message.get("event")
        if event in _MEMBER_SCOPE_FOR_ALL:
            user_id, scope = None, _MEMBER_SCOPE_FOR_ALL[event]
        elif event in _MEMBER_SCOPE_BY_EVENT:
            key, scope = _MEMBER_SCOPE_BY_EVENT[event]
            user_id = message.get(key)
            if not isinstance(user_id, int):
                return
        else:
            return

        async with self._lock:
            if user_id is None:
                targets = [(uid, ws) for uid, group in self._member_connections.items() for ws in group]
            else:
                targets = [(user_id, ws) for ws in self._member_connections.get(user_id, ())]
        if not targets:
            return

        payload = json.dumps({"event": "refresh", "scope": scope})
        for uid, ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                await self.disconnect_member(ws, uid)

    async def broadcast(self, message: dict[str, Any]):
        await self._notify_members(message)
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
