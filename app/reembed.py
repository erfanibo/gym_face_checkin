"""
Rebuilds face vectors after the recognizer model changed.

Vectors from different models are not comparable (see app/recognizer/base.py), so
after switching engine every existing member has NO usable sample until one is
made with the new model. The one thing that survives the switch is each member's
stored photo, so this job re-embeds that photo and stores the result as a new
sample (source='migration'). Old samples are left in place (a way back to the
previous engine). Safe to run any number of times: it only touches members that
have a photo but no sample of the active model.

The server runs it automatically at startup (FaceEngine), after the models are
loaded. By hand, e.g. to see what would happen first:

    python -m app.reembed --dry-run
    python -m app.reembed
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import config
from .database import db_cursor
from .face_samples import add_face_sample
from .recognizer import Recognizer


@dataclass
class ReembedReport:
    migrated: list[int] = field(default_factory=list)          # member ids that got a new sample
    no_photo_file: list[int] = field(default_factory=list)     # photo_path set, but the file is gone
    no_face: list[int] = field(default_factory=list)           # photo exists, no usable face in it
    failed: list[tuple[int, str]] = field(default_factory=list)  # unexpected errors (member id, message)
    pending: list[int] = field(default_factory=list)           # dry-run: members that WOULD be migrated

    @property
    def needs_attention(self) -> list[int]:
        """Members who still have no sample of the active model and need their face captured again."""
        return self.no_photo_file + self.no_face + [m for m, _ in self.failed]


def resolve_photo(photo_path: str | None) -> Path | None:
    """Photos are saved under DATA_DIR/static/*; the stored path may be from another machine/container,
    so look the file NAME up in the known folders rather than trusting the stored directory."""
    if not photo_path:
        return None
    name = Path(photo_path).name
    for folder in (config.PENDING_PHOTOS_DIR, config.USER_PHOTOS_DIR):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    direct = Path(photo_path)
    return direct if direct.is_file() else None


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def members_missing_model(model_id: str) -> list[dict]:
    """Members who have a stored photo but no sample made by `model_id`."""
    with db_cursor() as cur:
        cur.execute(
            """SELECT u.id, u.full_name, u.photo_path FROM registered_users u
               WHERE u.photo_path IS NOT NULL
                 AND NOT EXISTS (SELECT 1 FROM member_face_samples s WHERE s.user_id = u.id AND s.model = ?)
               ORDER BY u.id""",
            (model_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def reembed_missing_members(
    recognizer: Recognizer,
    dry_run: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> ReembedReport:
    report = ReembedReport()
    for member in members_missing_model(recognizer.model_id):
        if should_stop is not None and should_stop():
            break
        member_id = member["id"]
        photo = resolve_photo(member["photo_path"])
        if photo is None:
            report.no_photo_file.append(member_id)
            continue
        if dry_run:
            report.pending.append(member_id)
            continue
        try:
            image = read_image(photo)
            vector = recognizer.embed_image(image) if image is not None else None
            if vector is None:
                report.no_face.append(member_id)
                continue
            add_face_sample(member_id, vector, source="migration", model=recognizer.model_id)
            report.migrated.append(member_id)
        except Exception as exc:  # one broken photo must not stop the others
            report.failed.append((member_id, f"{type(exc).__name__}: {exc}"))
    return report


def describe(report: ReembedReport, model_id: str) -> str:
    parts = [f"model {model_id}: {len(report.migrated)} member(s) re-embedded from their photo"]
    if report.pending:
        parts.append(f"{len(report.pending)} would be re-embedded (dry run)")
    if report.needs_attention:
        parts.append(
            f"{len(report.needs_attention)} member(s) have NO usable face yet and must be captured again "
            f"(ids: {report.needs_attention})"
        )
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild member face vectors for the active recognizer model.")
    parser.add_argument("--dry-run", action="store_true", help="only list who would be re-embedded; write nothing")
    args = parser.parse_args(argv)

    from .database import init_db
    from .recognizer import get_recognizer

    init_db()
    recognizer = get_recognizer()
    if not args.dry_run:
        recognizer.warm_up()
    report = reembed_missing_members(recognizer, dry_run=args.dry_run)
    print(describe(report, recognizer.model_id))
    for member_id, message in report.failed:
        print(f"  member {member_id}: {message}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
