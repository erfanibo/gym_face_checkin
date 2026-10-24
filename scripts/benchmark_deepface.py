"""
Measure speed and accuracy of DeepFace models/detectors on THIS machine, with YOUR photos,
to choose DEEPFACE_MODEL / DEEPFACE_DETECTOR and the matching tolerance with numbers instead of guesses.

SPEED (every run): time of "find faces + embed them" for one frame, per model x detector.
    python scripts/benchmark_deepface.py --images photos/ --models SFace,Facenet512 --detectors yunet
    python scripts/benchmark_deepface.py --camera 0 --frames 30           # live frames from the webcam

ACCURACY (needs a folder of labelled photos, 2+ photos for each of 2+ people, ideally taken at the
real door/lighting, including glasses/angles that cause trouble today):
    photos/
        ali/   1.jpg 2.jpg 3.jpg
        sara/  1.jpg 2.jpg
It reports, per model: how far apart photos of the SAME person are vs DIFFERENT people, the best
cut-off, and -- using the tolerance the app would use right now -- how often a stranger would be let
in as someone else (false accept) and how often a member would not be recognised (false reject), plus a
leave-one-out "who is this?" test that mirrors what the door camera does.

Run it from the project folder (or: docker compose run --rm gym-face-checkin python scripts/benchmark_deepface.py ...).
A model that is not baked into the image downloads its weights the first time (needs internet once).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# pure evaluation logic (no DeepFace needed: unit-tested with fake vectors)
# --------------------------------------------------------------------------- #
def evaluate(
    gallery: dict[str, list[np.ndarray]],
    distances: Callable[[np.ndarray, np.ndarray], np.ndarray],
    known_tolerance: float,
) -> dict:
    """
    gallery: person name -> list of that person's vectors.
    distances(matrix, vec): distance from vec to every row of matrix (the recognizer's own).
    known_tolerance: the cut-off the app would use ("closer than this = same person").
    """
    people: list[str] = []
    vectors: list[np.ndarray] = []
    for person, vecs in gallery.items():
        for v in vecs:
            people.append(person)
            vectors.append(np.asarray(v, dtype=np.float64))
    n = len(vectors)
    if n < 2 or len(set(people)) < 2 or not any(people.count(p) >= 2 for p in set(people)):
        raise ValueError("need at least 2 people, and at least 2 photos of one of them")
    matrix = np.vstack(vectors)

    genuine: list[float] = []
    impostor: list[float] = []
    id_correct = id_wrong = id_unknown = 0
    for i in range(n):
        d = np.asarray(distances(matrix, matrix[i]), dtype=np.float64)
        for j in range(i + 1, n):
            (genuine if people[i] == people[j] else impostor).append(float(d[j]))
        if people.count(people[i]) >= 2:               # someone else of this person is in the gallery
            d_other = d.copy()
            d_other[i] = np.inf                         # leave this very photo out
            j = int(np.argmin(d_other))
            if d_other[j] <= known_tolerance:
                if people[j] == people[i]:
                    id_correct += 1
                else:
                    id_wrong += 1
            else:
                id_unknown += 1

    g, im = np.array(genuine), np.array(impostor)
    result: dict = {
        "photos": n,
        "people": len(set(people)),
        "genuine_pairs": len(g),
        "impostor_pairs": len(im),
        "genuine_mean": float(g.mean()) if len(g) else None,
        "impostor_mean": float(im.mean()) if len(im) else None,
        "false_accept_rate": float((im <= known_tolerance).mean()) if len(im) else None,
        "false_reject_rate": float((g > known_tolerance).mean()) if len(g) else None,
    }
    if len(g) and len(im):
        candidates = np.unique(np.concatenate([g, im]))
        best_thr, best_acc = float(candidates[0]), -1.0
        for thr in candidates:
            acc = ((g <= thr).mean() + (im > thr).mean()) / 2
            if acc > best_acc:
                best_thr, best_acc = float(thr), float(acc)
        result["best_threshold"] = best_thr
        result["best_balanced_accuracy"] = best_acc
    queries = id_correct + id_wrong + id_unknown
    result["identify"] = {
        "queries": queries,
        "correct": id_correct,
        "wrong_person": id_wrong,
        "not_recognised": id_unknown,
    }
    return result


def load_labelled_images(folder: Path) -> dict[str, list[Path]]:
    people: dict[str, list[Path]] = {}
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(f for f in sub.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            people[sub.name] = files
    return people


# --------------------------------------------------------------------------- #
# running it
# --------------------------------------------------------------------------- #
def read_image(path: Path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def time_frames(recognizer, frames: list[np.ndarray]) -> dict:
    recognizer.detect_and_embed(frames[0])           # first call includes lazy start-up work; not counted
    times = []
    found = 0
    for frame in frames:
        t = time.perf_counter()
        found += len(recognizer.detect_and_embed(frame))
        times.append((time.perf_counter() - t) * 1000)
    return {
        "frames": len(frames),
        "faces_found": found,
        "mean_ms": float(np.mean(times)),
        "p95_ms": percentile(times, 95),
        "max_ms": float(np.max(times)),
    }


def grab_camera_frames(index: int, count: int) -> list[np.ndarray]:
    import cv2
    video = cv2.VideoCapture(index)
    if not video.isOpened():
        raise SystemExit(f"cannot open camera {index}")
    frames = []
    try:
        for _ in range(count + 5):
            ok, frame = video.read()
            if ok and len(frames) < count and _ >= 5:       # skip the first few (auto-exposure settling)
                frames.append(frame)
            time.sleep(0.05)
    finally:
        video.release()
    if not frames:
        raise SystemExit("camera opened but gave no frames")
    return frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, help="folder of labelled photos: <folder>/<person>/<photo>.jpg")
    ap.add_argument("--camera", type=int, help="also time frames grabbed live from this camera index")
    ap.add_argument("--frames", type=int, default=30, help="frames to grab with --camera (default 30)")
    ap.add_argument("--models", default="SFace,Facenet512", help="comma separated DeepFace models")
    ap.add_argument("--detectors", default="yunet", help="comma separated DeepFace detectors")
    args = ap.parse_args(argv)
    if args.images is None and args.camera is None:
        ap.error("give --images and/or --camera")

    from app.recognizer.deepface_backend import DeepFaceRecognizer

    labelled = load_labelled_images(args.images) if args.images else {}
    images = {p: [read_image(f) for f in files] for p, files in labelled.items()}
    speed_frames = [img for imgs in images.values() for img in imgs if img is not None][:30]
    if args.camera is not None:
        speed_frames = grab_camera_frames(args.camera, args.frames) + speed_frames

    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        for detector in [d.strip() for d in args.detectors.split(",") if d.strip()]:
            label = f"{model} + {detector}"
            print(f"\n=== {label} ===", flush=True)
            try:
                rec = DeepFaceRecognizer(model_name=model, detector=detector)
                rec.warm_up()
            except Exception as exc:
                print(f"  cannot load: {type(exc).__name__}: {exc}")
                continue
            tol = rec.tolerances()
            print(f"  model standard threshold {rec.verification_threshold}  |  app would use: "
                  f"known<={tol.known}  same-unknown<={tol.pending}")
            row = {"label": label}
            if speed_frames:
                sp = time_frames(rec, speed_frames)
                row["speed"] = sp
                print(f"  speed: mean {sp['mean_ms']:.0f} ms  p95 {sp['p95_ms']:.0f} ms  max {sp['max_ms']:.0f} ms per frame "
                      f"({sp['frames']} frames, {sp['faces_found']} faces found)")
            if labelled:
                gallery: dict[str, list[np.ndarray]] = {}
                skipped = 0
                for person, imgs in images.items():
                    for img in imgs:
                        vec = rec.embed_image(img) if img is not None else None
                        if vec is None:
                            skipped += 1
                            continue
                        gallery.setdefault(person, []).append(vec)
                try:
                    ev = evaluate(gallery, rec.distances, tol.known)
                except ValueError as exc:
                    print(f"  accuracy: {exc}")
                else:
                    row["accuracy"] = ev
                    print(f"  accuracy on {ev['photos']} photos of {ev['people']} people"
                          + (f" ({skipped} unreadable photos skipped)" if skipped else "") + ":")
                    print(f"    same person   : average distance {ev['genuine_mean']:.3f}  ({ev['genuine_pairs']} pairs)")
                    print(f"    different     : average distance {ev['impostor_mean']:.3f}  ({ev['impostor_pairs']} pairs)")
                    print(f"    at known<={tol.known}: stranger accepted as member {ev['false_accept_rate']:.1%}, "
                          f"member not recognised {ev['false_reject_rate']:.1%}")
                    if "best_threshold" in ev:
                        print(f"    best cut-off on this data: {ev['best_threshold']:.3f}  "
                              f"(balanced accuracy {ev['best_balanced_accuracy']:.1%})")
                    idn = ev["identify"]
                    if idn["queries"]:
                        print(f"    'who is this?' (each photo vs all the others): correct {idn['correct']}/{idn['queries']}, "
                              f"WRONG person {idn['wrong_person']}, not recognised {idn['not_recognised']}")
            rows.append(row)

    print("\n=== summary ===")
    for row in rows:
        speed = row.get("speed")
        acc = row.get("accuracy")
        parts = [f"{row['label']:<24}"]
        if speed:
            parts.append(f"{speed['mean_ms']:>5.0f} ms/frame")
        if acc:
            idn = acc["identify"]
            parts.append(f"wrong-person {idn['wrong_person']}/{idn['queries']}  not-recognised {idn['not_recognised']}/{idn['queries']}")
        print("  " + "   ".join(parts))
    print("\nPick the model with ZERO wrong-person results first, then the fewest not-recognised, then the speed you can afford "
          "(the camera is processed about every 5th frame, so ~300 ms per frame is still fine for a door).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
