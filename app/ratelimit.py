"""
A tiny in-memory "too many failed attempts" limiter (one process, one gym -- no Redis needed).
Only FAILURES are counted; a successful login clears the counter for that key.
"""
import threading
import time


class FailureLimiter:
    def __init__(self, max_failures: int, window_seconds: int):
        self.max_failures = max_failures
        self.window = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _fresh(self, key: str, now: float) -> list[float]:
        hits = [t for t in self._hits.get(key, ()) if now - t < self.window]
        if hits:
            self._hits[key] = hits
        else:
            self._hits.pop(key, None)
        return hits

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._fresh(key, time.monotonic())) >= self.max_failures

    def record(self, key: str) -> None:
        with self._lock:
            now = time.monotonic()
            if len(self._hits) > 10_000:  # never grow without bound
                for k in list(self._hits):
                    self._fresh(k, now)
            self._hits.setdefault(key, []).append(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


# member panel: 8 wrong attempts per phone / 40 per client address, per 10 minutes
member_login_by_phone = FailureLimiter(8, 600)
member_login_by_ip = FailureLimiter(40, 600)
# changing an existing password needs the current one -- same budget per member
member_password_change = FailureLimiter(8, 600)

TOO_MANY_MESSAGE = "تعداد تلاش‌های ناموفق زیاد بود؛ چند دقیقه بعد دوباره تلاش کنید."
