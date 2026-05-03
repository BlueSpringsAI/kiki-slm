"""Ledger semantics + concurrency tests.

Most important: the atomic-claim test. If two workers both claim N records
they must get DISJOINT sets, never the same record twice. We verify this with
real OS threads against a real on-disk SQLite file (the Ledger's actual code
path) — not mocks."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts.loopper.v2.error_kinds import ErrorKind
from scripts.loopper.v2.ledger import (
    STATUS_DONE,
    STATUS_DROPPED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_QUARANTINED,
    Ledger,
    content_hash,
    new_run_id,
)


def test_init_creates_schema(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.db")
    # Re-opening is a no-op (no errors).
    ledger2 = Ledger(tmp_path / "l.db")
    ledger.close()
    ledger2.close()


def test_upsert_record_inserts(tmp_ledger):
    rec = tmp_ledger.upsert_record("T1", "refund_request", "trace", content_hash="h1")
    assert rec.ticket_id == "T1"
    assert rec.status == STATUS_PENDING
    assert rec.content_hash == "h1"
    assert rec.attempt_count == 0


def test_upsert_record_is_idempotent(tmp_ledger):
    r1 = tmp_ledger.upsert_record("T1", "refund_request", "trace", content_hash="h1")
    r2 = tmp_ledger.upsert_record("T1", "refund_request", "trace", content_hash="h1")
    assert r2.created_at == r1.created_at  # not re-inserted


def test_upsert_record_resets_when_content_changes(tmp_ledger):
    tmp_ledger.upsert_record("T1", "refund_request", "trace", content_hash="h1")
    rec = tmp_ledger.get_record("T1", "refund_request", "trace")
    # Mark it done first; new content should reset to pending.
    tmp_ledger.mark_done(rec, output_path="/tmp/out", run_id="r1")
    after_done = tmp_ledger.get_record("T1", "refund_request", "trace")
    assert after_done.status == STATUS_DONE

    tmp_ledger.upsert_record("T1", "refund_request", "trace", content_hash="h2")
    after_change = tmp_ledger.get_record("T1", "refund_request", "trace")
    assert after_change.status == STATUS_PENDING
    assert after_change.content_hash == "h2"
    assert after_change.error_kind is None


def test_claim_pending_basic(tmp_ledger):
    for tid in ("T1", "T2", "T3"):
        tmp_ledger.upsert_record(tid, "refund_request", "trace", content_hash="h")
    run_id = new_run_id("trace", "refund_request")
    claimed = tmp_ledger.claim_pending("refund_request", "trace", run_id=run_id, limit=2)
    assert len(claimed) == 2
    for rec in claimed:
        assert rec.status == STATUS_PROCESSING
        assert rec.attempt_count == 1
        assert rec.last_run_id == run_id


def test_claim_pending_returns_empty_when_no_pending(tmp_ledger):
    assert tmp_ledger.claim_pending("refund_request", "trace", run_id="r", limit=10) == []


def test_claim_pending_atomic_under_contention(tmp_path: Path):
    """Two threads each grab 50 records from a pool of 100. They must get
    disjoint sets — no record can be claimed twice."""
    db = tmp_path / "concurrency.db"
    ledger = Ledger(db)
    for i in range(100):
        ledger.upsert_record(f"T{i:03}", "refund_request", "trace", content_hash="h")

    barrier = threading.Barrier(2)
    results: dict[str, list[str]] = {}

    def claimer(tag: str):
        local_ledger = Ledger(db)  # separate connection per thread
        try:
            barrier.wait()
            claimed = local_ledger.claim_pending(
                "refund_request", "trace", run_id=tag, limit=50
            )
            results[tag] = [c.ticket_id for c in claimed]
        finally:
            local_ledger.close()

    with ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(claimer, "A")
        ex.submit(claimer, "B")

    a_ids = set(results["A"])
    b_ids = set(results["B"])
    overlap = a_ids & b_ids
    assert not overlap, f"two claimers got the same records: {overlap}"
    assert len(a_ids) + len(b_ids) == 100  # exact partition

    ledger.close()


def test_mark_done_transitions_status(tmp_ledger):
    tmp_ledger.upsert_record("T1", "refund_request", "trace", content_hash="h")
    [rec] = tmp_ledger.claim_pending("refund_request", "trace", run_id="r1", limit=1)
    tmp_ledger.mark_done(rec, output_path="/x/T1.json", run_id="r1")
    after = tmp_ledger.get_record("T1", "refund_request", "trace")
    assert after.status == STATUS_DONE
    assert after.output_path == "/x/T1.json"


def test_mark_quarantined_creates_entry_and_audit(tmp_ledger):
    tmp_ledger.upsert_record("T1", "refund_request", "trace", content_hash="h")
    [rec] = tmp_ledger.claim_pending("refund_request", "trace", run_id="r1", limit=1)
    entry = tmp_ledger.mark_quarantined(
        rec,
        error_kind=ErrorKind.TRACE_TEMPLATE_RESPONSE,
        error_msg="template detected",
        error_detail={"markers": ["i'll check"]},
        quarantine_file="/q/T1.json",
        run_id="r1",
    )
    after = tmp_ledger.get_record("T1", "refund_request", "trace")
    assert after.status == STATUS_QUARANTINED
    assert after.error_kind == ErrorKind.TRACE_TEMPLATE_RESPONSE.value
    assert entry.error_kind == ErrorKind.TRACE_TEMPLATE_RESPONSE.value
    assert entry.suggested_fix  # populated from META
    assert entry.detail["markers"] == ["i'll check"]
    history = tmp_ledger.record_history("T1", "refund_request")
    assert any(h["to_status"] == STATUS_QUARANTINED for h in history)


def test_resolve_quarantine_requeues_record(tmp_ledger):
    tmp_ledger.upsert_record("T1", "refund_request", "trace", content_hash="h")
    [rec] = tmp_ledger.claim_pending("refund_request", "trace", run_id="r1", limit=1)
    entry = tmp_ledger.mark_quarantined(
        rec, error_kind=ErrorKind.TRACE_TEMPLATE_RESPONSE,
        error_msg="msg", run_id="r1",
    )
    tmp_ledger.resolve_quarantine(entry, run_id="fixer-1")
    after = tmp_ledger.get_record("T1", "refund_request", "trace")
    assert after.status == STATUS_PENDING
    assert after.error_kind is None
    unresolved = tmp_ledger.unresolved_quarantine(error_kind=ErrorKind.TRACE_TEMPLATE_RESPONSE)
    assert len(unresolved) == 0


def test_resolve_quarantine_can_drop(tmp_ledger):
    tmp_ledger.upsert_record("T1", "refund_request", "trace", content_hash="h")
    [rec] = tmp_ledger.claim_pending("refund_request", "trace", run_id="r1", limit=1)
    entry = tmp_ledger.mark_quarantined(
        rec, error_kind=ErrorKind.TRACE_TEMPLATE_RESPONSE, error_msg="msg",
    )
    tmp_ledger.resolve_quarantine(entry, run_id="fixer-1", requeue_to=STATUS_DROPPED)
    after = tmp_ledger.get_record("T1", "refund_request", "trace")
    assert after.status == STATUS_DROPPED


def test_reset_orphans(tmp_ledger):
    """A record stuck in 'processing' beyond the cutoff should be reset to pending."""
    tmp_ledger.upsert_record("T1", "refund_request", "trace", content_hash="h")
    tmp_ledger.claim_pending("refund_request", "trace", run_id="r1", limit=1)
    # Manually backdate updated_at so it looks orphaned.
    with tmp_ledger._txn() as conn:
        conn.execute(
            "UPDATE records SET updated_at='2020-01-01T00:00:00+00:00' WHERE ticket_id='T1'"
        )
    n = tmp_ledger.reset_orphans(older_than_minutes=30)
    assert n == 1
    rec = tmp_ledger.get_record("T1", "refund_request", "trace")
    assert rec.status == STATUS_PENDING


def test_run_lifecycle(tmp_ledger):
    rid = new_run_id("trace", "refund_request")
    run = tmp_ledger.start_run(
        run_id=rid, stage="trace", intent="refund_request",
        args={"concurrency": 5}, git_commit="abc12345",
    )
    assert run.status == "running"
    tmp_ledger.finish_run(rid, status="done", processed=10, done=8, quarantined=2)
    [r] = [r for r in tmp_ledger.recent_runs(50) if r.run_id == rid]
    assert r.status == "done"
    assert r.records_quarantined == 2
    assert r.finished_at is not None


def test_counts_matrix(tmp_ledger):
    for i, status in enumerate([STATUS_DONE, STATUS_DONE, STATUS_PENDING, STATUS_QUARANTINED]):
        tmp_ledger.upsert_record(f"T{i}", "refund_request", "trace", content_hash="h")
        if status != STATUS_PENDING:
            [rec] = tmp_ledger.claim_pending("refund_request", "trace", run_id="r", limit=1)
            if status == STATUS_DONE:
                tmp_ledger.mark_done(rec, output_path="/x", run_id="r")
            elif status == STATUS_QUARANTINED:
                tmp_ledger.mark_quarantined(
                    rec, error_kind=ErrorKind.TRACE_TEMPLATE_RESPONSE, error_msg="x"
                )
    rows = tmp_ledger.counts_matrix()
    by_status = {r["status"]: r["n"] for r in rows if r["intent"] == "refund_request"}
    assert by_status[STATUS_DONE] == 2
    assert by_status[STATUS_PENDING] == 1
    assert by_status[STATUS_QUARANTINED] == 1


def test_quarantine_breakdown(tmp_ledger):
    for tid in ("T1", "T2"):
        tmp_ledger.upsert_record(tid, "refund_request", "trace", content_hash="h")
        [rec] = tmp_ledger.claim_pending("refund_request", "trace", run_id="r", limit=1)
        tmp_ledger.mark_quarantined(
            rec, error_kind=ErrorKind.TRACE_TEMPLATE_RESPONSE, error_msg="x"
        )
    breakdown = tmp_ledger.quarantine_breakdown()
    [row] = [b for b in breakdown if b["error_kind"] == ErrorKind.TRACE_TEMPLATE_RESPONSE.value]
    assert row["unresolved"] == 2


def test_content_hash_stable():
    a = content_hash({"a": 1, "b": 2})
    b = content_hash({"b": 2, "a": 1})
    assert a == b  # key order should not matter
    assert content_hash("x") == content_hash(b"x")
