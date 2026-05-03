"""SQLite-backed state ledger — the single source of truth for v2 pipeline state.

Every stage script and every fixer reads from / writes to the ledger. Outside
of the ledger there is no authoritative state (artifact files are derivable;
the ledger tells you what artifacts SHOULD exist).

Tables
------
records           one row per (ticket_id, intent, stage); state machine
runs              one row per script invocation; immutable after finish
quarantine_entries  one row per quarantined record; carries fix details
audit_log         append-only history of every state transition
schema_version    migration guard

Concurrency model
-----------------
- WAL mode + busy_timeout=5000ms — many readers, one writer at a time
- claim_pending() uses BEGIN IMMEDIATE to atomically grab N pending records
  and flip them to 'processing' in one transaction. No two workers can claim
  the same record.
- Stage shard locks (`flock` on data/v2/locks/{intent}.{stage}.lock) provide
  the OUTER guarantee that only one worker process per (intent, stage) is
  active. The atomic claim handles inner concurrency among threads/coroutines
  within that one process.

Design choices
--------------
- Composite primary key (ticket_id, intent, stage). A ticket has exactly 7
  rows, one per stage. Stage transitions are explicit row inserts/updates,
  never schema-implicit.
- No FK enforcement (PRAGMA foreign_keys=OFF default). We manage referential
  integrity in code; FKs would require ON DELETE rules we don't want.
- Timestamps as ISO strings (not native datetimes) so dashboards can read
  them without deserialization.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from scripts.loopper.v2.error_kinds import ErrorKind, Stage, meta

SCHEMA_VERSION = 1


# ── Status constants (kept as plain strings so they round-trip cleanly through SQLite) ──

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_QUARANTINED = "quarantined"
STATUS_DROPPED = "dropped"

ALL_STATUSES = (STATUS_PENDING, STATUS_PROCESSING, STATUS_DONE, STATUS_QUARANTINED, STATUS_DROPPED)
TERMINAL_STATUSES = (STATUS_DONE, STATUS_DROPPED)


# ── Schema ──

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS records (
    ticket_id     TEXT NOT NULL,
    intent        TEXT NOT NULL,
    stage         TEXT NOT NULL,
    status        TEXT NOT NULL,
    content_hash  TEXT,
    output_path   TEXT,
    error_kind    TEXT,
    error_msg     TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_run_id   TEXT,
    last_worker   TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (ticket_id, intent, stage)
);

CREATE INDEX IF NOT EXISTS idx_records_pending
    ON records(intent, stage, status);

CREATE INDEX IF NOT EXISTS idx_records_quarantined
    ON records(error_kind) WHERE status = 'quarantined';

CREATE INDEX IF NOT EXISTS idx_records_processing
    ON records(updated_at) WHERE status = 'processing';

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    intent              TEXT,
    stage               TEXT,
    worker              TEXT,
    git_commit          TEXT,
    args_json           TEXT,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL,           -- running|done|failed|aborted
    records_processed   INTEGER NOT NULL DEFAULT 0,
    records_done        INTEGER NOT NULL DEFAULT 0,
    records_quarantined INTEGER NOT NULL DEFAULT 0,
    records_dropped     INTEGER NOT NULL DEFAULT 0,
    records_failed      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_runs_active
    ON runs(intent, stage, status);

CREATE TABLE IF NOT EXISTS quarantine_entries (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id          TEXT NOT NULL,
    intent             TEXT NOT NULL,
    stage              TEXT NOT NULL,
    error_kind         TEXT NOT NULL,
    error_msg          TEXT,
    error_detail_json  TEXT,
    suggested_fix      TEXT,
    quarantine_file    TEXT,
    attempts           INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL,
    resolved_at        TEXT,
    resolved_by_run_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_quar_unresolved
    ON quarantine_entries(error_kind) WHERE resolved_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_quar_record
    ON quarantine_entries(ticket_id, intent, stage);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   TEXT NOT NULL,
    intent      TEXT NOT NULL,
    stage       TEXT NOT NULL,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    error_kind  TEXT,
    run_id      TEXT,
    note        TEXT,
    ts          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_record
    ON audit_log(ticket_id, intent, stage, ts);
"""


# ── Dataclasses ──

@dataclass
class Record:
    ticket_id: str
    intent: str
    stage: str
    status: str
    content_hash: str | None = None
    output_path: str | None = None
    error_kind: str | None = None
    error_msg: str | None = None
    attempt_count: int = 0
    last_run_id: str | None = None
    last_worker: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Run:
    run_id: str
    intent: str | None
    stage: str | None
    worker: str
    git_commit: str | None
    args_json: str | None
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    records_processed: int = 0
    records_done: int = 0
    records_quarantined: int = 0
    records_dropped: int = 0
    records_failed: int = 0


@dataclass
class QuarantineEntry:
    id: int
    ticket_id: str
    intent: str
    stage: str
    error_kind: str
    error_msg: str | None
    error_detail_json: str | None
    suggested_fix: str | None
    quarantine_file: str | None
    attempts: int
    created_at: str
    resolved_at: str | None
    resolved_by_run_id: str | None

    @property
    def detail(self) -> dict[str, Any]:
        if not self.error_detail_json:
            return {}
        return json.loads(self.error_detail_json)


# ── Helpers ──

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def worker_id() -> str:
    """Stable identifier for the current process — used to attribute claims."""
    return f"{os.environ.get('USER', 'unknown')}@{socket.gethostname()}.{os.getpid()}"


def content_hash(payload: bytes | str | dict) -> str:
    """Stable SHA-256 of any input. Used for idempotency: re-running on the same input is a no-op."""
    if isinstance(payload, dict):
        payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_run_id(stage: str, intent: str | None = None) -> str:
    """Generate a run id: ISO-ish prefix + intent + stage + short uuid."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short = uuid.uuid4().hex[:8]
    if intent:
        return f"{ts}_{intent}_{stage}_{short}"
    return f"{ts}_{stage}_{short}"


# ── Ledger ──

class Ledger:
    """Thin wrapper around a SQLite connection. Thread-safe via a single connection per Ledger
    instance plus an internal lock; for multi-process use, each process should construct its own."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because we serialize writes ourselves via _lock.
        # isolation_level=None gives us manual BEGIN/COMMIT control.
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._setup()

    # ── lifecycle ──

    def _setup(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._conn.executescript(_SCHEMA_SQL)
            row = self._conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )
            elif row["version"] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Ledger schema version mismatch: db={row['version']} code={SCHEMA_VERSION}. "
                    "Run a migration before continuing."
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextlib.contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE / COMMIT, with rollback on exception. Holds the writer lock."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE;")
            try:
                yield self._conn
                self._conn.execute("COMMIT;")
            except Exception:
                self._conn.execute("ROLLBACK;")
                raise

    # ── records ──

    def upsert_record(
        self,
        ticket_id: str,
        intent: str,
        stage: str,
        *,
        status: str = STATUS_PENDING,
        content_hash: str | None = None,
        run_id: str | None = None,
    ) -> Record:
        """Insert a new record or no-op if one exists with same content_hash.

        If a record exists with a DIFFERENT content_hash, downstream stages of that
        ticket should be invalidated by the caller (typically by upstream stages). We
        don't cascade automatically — explicit is better than surprise wipes.
        """
        now = _now()
        with self._txn() as conn:
            existing = conn.execute(
                "SELECT * FROM records WHERE ticket_id=? AND intent=? AND stage=?",
                (ticket_id, intent, stage),
            ).fetchone()

            if existing is None:
                conn.execute(
                    """INSERT INTO records
                       (ticket_id, intent, stage, status, content_hash,
                        attempt_count, last_run_id, last_worker, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)""",
                    (ticket_id, intent, stage, status, content_hash,
                     run_id, worker_id(), now, now),
                )
                self._audit(conn, ticket_id, intent, stage, None, status, run_id, "insert")
            elif content_hash and existing["content_hash"] != content_hash:
                # Input changed — re-set to pending so downstream re-processes.
                conn.execute(
                    """UPDATE records
                       SET status=?, content_hash=?, error_kind=NULL, error_msg=NULL,
                           output_path=NULL, last_run_id=?, last_worker=?, updated_at=?
                       WHERE ticket_id=? AND intent=? AND stage=?""",
                    (STATUS_PENDING, content_hash, run_id, worker_id(), now,
                     ticket_id, intent, stage),
                )
                self._audit(conn, ticket_id, intent, stage, existing["status"], STATUS_PENDING,
                            run_id, "input_changed")

        return self.get_record(ticket_id, intent, stage)  # type: ignore[return-value]

    def get_record(self, ticket_id: str, intent: str, stage: str) -> Record | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM records WHERE ticket_id=? AND intent=? AND stage=?",
                (ticket_id, intent, stage),
            ).fetchone()
        return _row_to_record(row) if row else None

    def claim_pending(
        self,
        intent: str,
        stage: str,
        *,
        run_id: str,
        limit: int = 1,
    ) -> list[Record]:
        """Atomically claim up to `limit` pending records. Flips them to 'processing'.

        Claimed records are guaranteed not to be picked up by any other claim_pending
        call. The caller MUST call mark_done / mark_quarantined / mark_dropped on each
        claimed record, or call requeue() to release them.
        """
        now = _now()
        worker = worker_id()
        with self._txn() as conn:
            rows = conn.execute(
                """SELECT ticket_id FROM records
                   WHERE intent=? AND stage=? AND status=?
                   ORDER BY updated_at LIMIT ?""",
                (intent, stage, STATUS_PENDING, limit),
            ).fetchall()
            if not rows:
                return []
            ids = [r["ticket_id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"""UPDATE records
                    SET status=?, last_run_id=?, last_worker=?, updated_at=?,
                        attempt_count=attempt_count+1
                    WHERE intent=? AND stage=? AND ticket_id IN ({placeholders})""",
                (STATUS_PROCESSING, run_id, worker, now, intent, stage, *ids),
            )
            for tid in ids:
                self._audit(conn, tid, intent, stage, STATUS_PENDING, STATUS_PROCESSING, run_id, "claim")
            claimed = conn.execute(
                f"""SELECT * FROM records
                    WHERE intent=? AND stage=? AND ticket_id IN ({placeholders})""",
                (intent, stage, *ids),
            ).fetchall()
        return [_row_to_record(r) for r in claimed]

    def mark_done(
        self,
        record: Record,
        *,
        output_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        now = _now()
        with self._txn() as conn:
            conn.execute(
                """UPDATE records
                   SET status=?, output_path=?, error_kind=NULL, error_msg=NULL,
                       last_run_id=?, updated_at=?
                   WHERE ticket_id=? AND intent=? AND stage=?""",
                (STATUS_DONE, output_path, run_id, now,
                 record.ticket_id, record.intent, record.stage),
            )
            self._audit(conn, record.ticket_id, record.intent, record.stage,
                        record.status, STATUS_DONE, run_id, None)

    def mark_quarantined(
        self,
        record: Record,
        *,
        error_kind: ErrorKind,
        error_msg: str,
        error_detail: dict[str, Any] | None = None,
        quarantine_file: str | None = None,
        run_id: str | None = None,
    ) -> QuarantineEntry:
        """Flip record to quarantined and enqueue a quarantine_entries row in the same txn."""
        now = _now()
        kind_meta = meta(error_kind)
        suggested_fix = kind_meta.suggested_fix
        detail_json = json.dumps(error_detail) if error_detail else None
        with self._txn() as conn:
            conn.execute(
                """UPDATE records
                   SET status=?, error_kind=?, error_msg=?, last_run_id=?, updated_at=?
                   WHERE ticket_id=? AND intent=? AND stage=?""",
                (STATUS_QUARANTINED, error_kind.value, error_msg, run_id, now,
                 record.ticket_id, record.intent, record.stage),
            )
            cur = conn.execute(
                """INSERT INTO quarantine_entries
                   (ticket_id, intent, stage, error_kind, error_msg, error_detail_json,
                    suggested_fix, quarantine_file, attempts, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (record.ticket_id, record.intent, record.stage, error_kind.value,
                 error_msg, detail_json, suggested_fix, quarantine_file, now),
            )
            entry_id = cur.lastrowid
            self._audit(conn, record.ticket_id, record.intent, record.stage,
                        record.status, STATUS_QUARANTINED, run_id, error_kind.value)
            row = conn.execute(
                "SELECT * FROM quarantine_entries WHERE id=?", (entry_id,)
            ).fetchone()
        return _row_to_quarantine(row)

    def mark_dropped(
        self,
        record: Record,
        *,
        error_kind: ErrorKind,
        reason: str,
        run_id: str | None = None,
    ) -> None:
        now = _now()
        with self._txn() as conn:
            conn.execute(
                """UPDATE records
                   SET status=?, error_kind=?, error_msg=?, last_run_id=?, updated_at=?
                   WHERE ticket_id=? AND intent=? AND stage=?""",
                (STATUS_DROPPED, error_kind.value, reason, run_id, now,
                 record.ticket_id, record.intent, record.stage),
            )
            self._audit(conn, record.ticket_id, record.intent, record.stage,
                        record.status, STATUS_DROPPED, run_id, error_kind.value)

    def requeue(
        self,
        ticket_id: str,
        intent: str,
        stage: str,
        *,
        reason: str = "manual",
        run_id: str | None = None,
    ) -> None:
        """Reset a record to pending. Used by fixers after applying a fix."""
        now = _now()
        with self._txn() as conn:
            row = conn.execute(
                "SELECT status FROM records WHERE ticket_id=? AND intent=? AND stage=?",
                (ticket_id, intent, stage),
            ).fetchone()
            if not row:
                raise KeyError(f"no record for ({ticket_id}, {intent}, {stage})")
            conn.execute(
                """UPDATE records
                   SET status=?, error_kind=NULL, error_msg=NULL, last_run_id=?, updated_at=?
                   WHERE ticket_id=? AND intent=? AND stage=?""",
                (STATUS_PENDING, run_id, now, ticket_id, intent, stage),
            )
            self._audit(conn, ticket_id, intent, stage, row["status"], STATUS_PENDING, run_id, reason)

    def reset_orphans(self, *, older_than_minutes: int = 30) -> int:
        """Reset records stuck in 'processing' for too long (crashed workers).
        Returns count reset."""
        cutoff = (datetime.now(timezone.utc).timestamp() - older_than_minutes * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat(timespec="seconds")
        now = _now()
        with self._txn() as conn:
            rows = conn.execute(
                """SELECT ticket_id, intent, stage FROM records
                   WHERE status=? AND updated_at < ?""",
                (STATUS_PROCESSING, cutoff_iso),
            ).fetchall()
            if not rows:
                return 0
            for r in rows:
                conn.execute(
                    """UPDATE records SET status=?, updated_at=?
                       WHERE ticket_id=? AND intent=? AND stage=?""",
                    (STATUS_PENDING, now, r["ticket_id"], r["intent"], r["stage"]),
                )
                self._audit(conn, r["ticket_id"], r["intent"], r["stage"],
                            STATUS_PROCESSING, STATUS_PENDING, None, "orphan_reset")
        return len(rows)

    # ── runs ──

    def start_run(
        self,
        *,
        run_id: str,
        stage: str | None,
        intent: str | None,
        args: dict | None = None,
        git_commit: str | None = None,
    ) -> Run:
        now = _now()
        run = Run(
            run_id=run_id,
            intent=intent,
            stage=stage,
            worker=worker_id(),
            git_commit=git_commit,
            args_json=json.dumps(args) if args else None,
            started_at=now,
            status="running",
        )
        with self._txn() as conn:
            conn.execute(
                """INSERT INTO runs
                   (run_id, intent, stage, worker, git_commit, args_json, started_at, status,
                    records_processed, records_done, records_quarantined, records_dropped, records_failed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0)""",
                (run.run_id, run.intent, run.stage, run.worker, run.git_commit,
                 run.args_json, run.started_at, run.status),
            )
        return run

    def finish_run(
        self,
        run_id: str,
        *,
        status: str = "done",
        processed: int = 0,
        done: int = 0,
        quarantined: int = 0,
        dropped: int = 0,
        failed: int = 0,
    ) -> None:
        now = _now()
        with self._txn() as conn:
            conn.execute(
                """UPDATE runs
                   SET finished_at=?, status=?,
                       records_processed=?, records_done=?, records_quarantined=?,
                       records_dropped=?, records_failed=?
                   WHERE run_id=?""",
                (now, status, processed, done, quarantined, dropped, failed, run_id),
            )

    # ── quarantine ──

    def unresolved_quarantine(
        self,
        *,
        error_kind: ErrorKind | None = None,
        intent: str | None = None,
        stage: str | None = None,
        max_attempts: int | None = None,
        limit: int = 100,
    ) -> list[QuarantineEntry]:
        clauses = ["resolved_at IS NULL"]
        params: list[Any] = []
        if error_kind:
            clauses.append("error_kind=?")
            params.append(error_kind.value)
        if intent:
            clauses.append("intent=?")
            params.append(intent)
        if stage:
            clauses.append("stage=?")
            params.append(stage)
        if max_attempts is not None:
            clauses.append("attempts <= ?")
            params.append(max_attempts)
        sql = f"""SELECT * FROM quarantine_entries
                  WHERE {' AND '.join(clauses)}
                  ORDER BY created_at LIMIT ?"""
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_quarantine(r) for r in rows]

    def resolve_quarantine(
        self,
        entry: QuarantineEntry,
        *,
        run_id: str,
        requeue_to: str = STATUS_PENDING,
    ) -> None:
        """Mark a quarantine entry resolved and (by default) flip the record to pending so
        the main pipeline picks it up. Pass requeue_to=STATUS_DROPPED to give up."""
        now = _now()
        with self._txn() as conn:
            conn.execute(
                """UPDATE quarantine_entries
                   SET resolved_at=?, resolved_by_run_id=?
                   WHERE id=?""",
                (now, run_id, entry.id),
            )
            conn.execute(
                """UPDATE records
                   SET status=?, error_kind=NULL, error_msg=NULL, last_run_id=?, updated_at=?
                   WHERE ticket_id=? AND intent=? AND stage=?""",
                (requeue_to, run_id, now, entry.ticket_id, entry.intent, entry.stage),
            )
            self._audit(conn, entry.ticket_id, entry.intent, entry.stage,
                        STATUS_QUARANTINED, requeue_to, run_id, f"resolved:{entry.error_kind}")

    def bump_quarantine_attempts(self, entry_id: int) -> None:
        with self._txn() as conn:
            conn.execute(
                "UPDATE quarantine_entries SET attempts = attempts + 1 WHERE id=?",
                (entry_id,),
            )

    # ── queries (for status/dashboard) ──

    def counts_matrix(self) -> list[dict[str, Any]]:
        """Per-intent × per-stage × per-status counts. Drives `make status`."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT intent, stage, status, COUNT(*) AS n
                   FROM records GROUP BY intent, stage, status"""
            ).fetchall()
        return [dict(r) for r in rows]

    def quarantine_breakdown(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT intent, stage, error_kind, COUNT(*) AS unresolved
                   FROM quarantine_entries
                   WHERE resolved_at IS NULL
                   GROUP BY intent, stage, error_kind"""
            ).fetchall()
        return [dict(r) for r in rows]

    def active_runs(self) -> list[Run]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE status='running' ORDER BY started_at"
            ).fetchall()
        return [_row_to_run(r) for r in rows]

    def recent_runs(self, limit: int = 50) -> list[Run]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_run(r) for r in rows]

    def record_history(self, ticket_id: str, intent: str | None = None) -> list[dict[str, Any]]:
        sql = ("SELECT * FROM audit_log WHERE ticket_id=?"
               + (" AND intent=?" if intent else "")
               + " ORDER BY ts")
        params: tuple = (ticket_id, intent) if intent else (ticket_id,)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def stage_counts(self, intent: str, stage: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT status, COUNT(*) AS n FROM records
                   WHERE intent=? AND stage=? GROUP BY status""",
                (intent, stage),
            ).fetchall()
        out = {s: 0 for s in ALL_STATUSES}
        for r in rows:
            out[r["status"]] = r["n"]
        return out

    # ── audit ──

    def _audit(
        self,
        conn: sqlite3.Connection,
        ticket_id: str,
        intent: str,
        stage: str,
        from_status: str | None,
        to_status: str,
        run_id: str | None,
        note: str | None,
    ) -> None:
        conn.execute(
            """INSERT INTO audit_log
               (ticket_id, intent, stage, from_status, to_status, error_kind, run_id, note, ts)
               VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)""",
            (ticket_id, intent, stage, from_status, to_status, run_id, note, _now()),
        )


# ── Row mappers ──

def _row_to_record(row: sqlite3.Row | None) -> Record | None:
    if row is None:
        return None
    return Record(
        ticket_id=row["ticket_id"],
        intent=row["intent"],
        stage=row["stage"],
        status=row["status"],
        content_hash=row["content_hash"],
        output_path=row["output_path"],
        error_kind=row["error_kind"],
        error_msg=row["error_msg"],
        attempt_count=row["attempt_count"],
        last_run_id=row["last_run_id"],
        last_worker=row["last_worker"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        run_id=row["run_id"],
        intent=row["intent"],
        stage=row["stage"],
        worker=row["worker"],
        git_commit=row["git_commit"],
        args_json=row["args_json"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        status=row["status"],
        records_processed=row["records_processed"],
        records_done=row["records_done"],
        records_quarantined=row["records_quarantined"],
        records_dropped=row["records_dropped"],
        records_failed=row["records_failed"],
    )


def _row_to_quarantine(row: sqlite3.Row) -> QuarantineEntry:
    return QuarantineEntry(
        id=row["id"],
        ticket_id=row["ticket_id"],
        intent=row["intent"],
        stage=row["stage"],
        error_kind=row["error_kind"],
        error_msg=row["error_msg"],
        error_detail_json=row["error_detail_json"],
        suggested_fix=row["suggested_fix"],
        quarantine_file=row["quarantine_file"],
        attempts=row["attempts"],
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
        resolved_by_run_id=row["resolved_by_run_id"],
    )
