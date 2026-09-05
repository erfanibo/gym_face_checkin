"""
Backup / restore for the Settings tab ("پشتیبان‌گیری").

A backup is a single .zip containing:
  - gym_face.db   (a consistent online-backup snapshot, see database.backup_database_to)
  - static/       (member photos + pending-queue photos, so a restore doesn't
                    leave the panel showing broken image icons)

Restore is intentionally destructive-with-a-safety-net: it REPLACES the live
database and photo folders with whatever is inside the uploaded zip. Before
touching anything, the current .db file is copied aside
(gym_face.pre_restore_<timestamp>.db, next to the real one) so a bad/wrong
backup can still be recovered from manually -- this endpoint does not ask
for confirmation itself, that's the frontend's job (a plain browser confirm()
before the upload is even sent).

There is no authentication anywhere else in this app (see the rest of
app/routers/), so this endpoint doesn't add any either -- consistent with the
rest of the API, not a gap specific to backups. Same physical/network trust
model as everything else here.
"""
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .. import config
from ..database import backup_database_to, close_connection, init_db
from ..ws_manager import manager

router = APIRouter(prefix="/api/backup", tags=["backup"])


@router.get("")
def download_backup():
    tmp_dir = Path(tempfile.mkdtemp(prefix="gym_backup_"))
    db_snapshot = tmp_dir / "gym_face.db"
    backup_database_to(db_snapshot)  # consistent snapshot, not a raw file copy

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_path = tmp_dir / f"gym_backup_{stamp}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(db_snapshot, arcname="gym_face.db")
        if config.STATIC_DIR.exists():
            for path in config.STATIC_DIR.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(Path("static") / path.relative_to(config.STATIC_DIR)))

    # tmp_dir (including the raw db snapshot and the zip itself) is deleted
    # only AFTER the response has finished sending, via BackgroundTask
    return FileResponse(
        path=zip_path,
        filename=zip_path.name,
        media_type="application/zip",
        background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
    )


@router.post("/restore")
async def restore_backup(request: Request, file: UploadFile):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "فایل بک‌آپ باید یک فایل zip باشد")

    tmp_dir = Path(tempfile.mkdtemp(prefix="gym_restore_"))
    try:
        upload_path = tmp_dir / "uploaded.zip"
        with open(upload_path, "wb") as f:
            f.write(await file.read())

        extract_dir = tmp_dir / "extracted"
        try:
            with zipfile.ZipFile(upload_path) as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(400, "فایل بک‌آپ خراب است یا یک فایل zip معتبر نیست")

        restored_db = extract_dir / "gym_face.db"
        if not restored_db.exists():
            raise HTTPException(400, "این فایل یک بک‌آپ معتبر نیست (gym_face.db در آن پیدا نشد)")

        # --- from here on we start actually touching live state ---
        engine = request.app.state.face_engine
        engine.stop()  # releases the camera + stops writing to the DB
        close_connection()  # release the OS-level handle on the live .db file

        # safety net: keep the pre-restore db so a wrong/bad backup is recoverable
        if config.DB_PATH.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safety_copy = config.DB_PATH.with_name(f"gym_face.pre_restore_{stamp}.db")
            shutil.copy2(config.DB_PATH, safety_copy)

        shutil.copy2(restored_db, config.DB_PATH)

        # photos are optional in a backup (older/manual backups may not have
        # them) -- restore whatever categories are actually present, leave
        # the rest of the current static/ folder untouched otherwise
        restored_static = extract_dir / "static"
        for subfolder in ("user_photos", "pending_faces"):
            src = restored_static / subfolder
            if src.exists():
                dest = config.STATIC_DIR / subfolder
                shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(src, dest)

        init_db()  # reopens the connection + runs migrations against the restored file
        engine.reload_all()
        engine.start()

        with_photos = (extract_dir / "static").exists()
        await manager.broadcast({"event": "backup_restored"})

        return {"ok": True, "restored_photos": with_photos}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
