import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .database import init_db
from .face_engine import FaceEngine
from .routers import queue, users
from .ws_manager import manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    loop = asyncio.get_running_loop()
    engine = FaceEngine(loop)
    app.state.face_engine = engine
    engine.start()

    yield

    engine.stop()


app = FastAPI(title="Gym Face Check-in", lifespan=lifespan)

# Wide-open CORS for local development; tighten this once the panel has a
# fixed origin (e.g. served from the same FastAPI instance, as done below).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
app.mount("/panel", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="panel")

app.include_router(queue.router)
app.include_router(users.router)


@app.websocket("/ws/queue")
async def ws_queue(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # The panel doesn't need to send anything; we just keep the
            # socket open and wait for a disconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@app.get("/health")
def health():
    return {"status": "ok"}
