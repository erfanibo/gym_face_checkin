"""
Registering a membership plan for a member -- the one place this logic lives, shared by
every entry point (routers/users.py: "assign plan" + manual add; routers/queue.py:
camera registration), so they can never drift apart.

Same date rule as the months-based renewal: if the member still has time left, the plan's
days are added ON TOP of the current expiry (renewing early never costs days); if they are
expired or never had an expiry, the period starts now.
"""
from datetime import datetime, timedelta, timezone


def get_active_plan(cur, plan_id: int):
    cur.execute("SELECT * FROM membership_plans WHERE id=? AND is_active=1", (plan_id,))
    return cur.fetchone()


def apply_plan(cur, user_id: int, plan, now: datetime | None = None) -> str:
    """Extends the member's expiry by plan['duration_days'], records a purchase snapshot, and
    returns the new expiry as a UTC ISO string. Runs on the CALLER's cursor so it joins the
    caller's transaction (user + plan are saved together or not at all)."""
    now = now or datetime.now(timezone.utc)
    cur.execute("SELECT membership_expires_at FROM registered_users WHERE id=?", (user_id,))
    row = cur.fetchone()

    current_expiry = None
    if row and row["membership_expires_at"]:
        try:
            current_expiry = datetime.fromisoformat(row["membership_expires_at"])
        except ValueError:
            current_expiry = None

    base = current_expiry if (current_expiry and current_expiry > now) else now
    new_expiry = base + timedelta(days=plan["duration_days"])

    cur.execute(
        """INSERT INTO membership_purchases
               (user_id, plan_id, plan_name, price, duration_days, starts_at, expires_at, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, plan["id"], plan["name"], plan["price"], plan["duration_days"],
         base.isoformat(), new_expiry.isoformat(), now.isoformat()),
    )
    cur.execute("UPDATE registered_users SET membership_expires_at=? WHERE id=?", (new_expiry.isoformat(), user_id))
    return new_expiry.isoformat()
