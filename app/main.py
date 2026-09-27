import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles

from . import auth, config
from .database import init_db
from .face_engine import FaceEngine
from .routers import auth as auth_router
from .routers import backup, logs, queue, users
from .ws_manager import manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    auth.ensure_manager_password_seeded()

    loop = asyncio.get_running_loop()
    engine = FaceEngine(loop)
    app.state.face_engine = engine
    engine.start()

    yield

    engine.stop()


app = FastAPI(title="Gym Face Check-in", lifespan=lifespan)

# Session cookie (login state) -- must be added before any route that reads
# request.session. NOTE: allow_credentials below requires the panel to be
# served from the SAME origin as the API (it is, via the /panel mount below),
# since "*" + credentials is rejected by browsers.
app.add_middleware(SessionMiddleware, secret_key=config.SESSION_SECRET_KEY, max_age=config.SESSION_MAX_AGE_SECONDS)

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

app.include_router(auth_router.router)
app.include_router(queue.router)
app.include_router(users.router)
app.include_router(logs.router)
app.include_router(backup.router)


@app.websocket("/ws/queue")
async def ws_queue(websocket: WebSocket):
    # SessionMiddleware also populates .session on the websocket handshake
    # (cookies are sent with the upgrade request), so the same login applies.
    if websocket.session.get("role") != "manager":
        await websocket.close(code=4401)
        return
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
