"""Taxonomy invariants. These tests are the contract enforcement for the
ErrorKind enum — adding a member without metadata fails the suite."""
from __future__ import annotations

from scripts.loopper.v2.error_kinds import (
    META,
    ErrorKind,
    Severity,
    Stage,
    fixable_kinds,
    kinds_for_stage,
    meta,
)


def test_every_error_kind_has_metadata():
    """Every ErrorKind member must have a META entry. No orphans."""
    missing = [k for k in ErrorKind if k not in META]
    assert not missing, f"ErrorKind members without META entry: {missing}"


def test_no_orphan_meta_entries():
    """META can't reference an ErrorKind that doesn't exist."""
    extra = [k for k in META if k not in set(ErrorKind)]
    assert not extra, f"META entries without ErrorKind member: {extra}"


def test_meta_helper_returns_entry():
    m = meta(ErrorKind.RAW_TOO_SHORT)
    assert m.stage == Stage.SOURCE
    assert m.severity == Severity.DROP
    assert m.suggested_fix


def test_kinds_for_stage_partitions_correctly():
    seen: set[ErrorKind] = set()
    for stage in Stage.ordered():
        for k in kinds_for_stage(stage):
            assert k not in seen, f"{k} appears in multiple stages"
            seen.add(k)
    # everything in META should belong to exactly one stage
    assert seen == set(META.keys())


def test_fixable_kinds_excludes_drops_and_aggregates():
    fix = set(fixable_kinds())
    for k in ErrorKind:
        sev = META[k].severity
        if sev in (Severity.QUARANTINE, Severity.FIXABLE):
            assert k in fix, f"{k} ({sev}) should be in fixable_kinds()"
        else:
            assert k not in fix, f"{k} ({sev}) should not be in fixable_kinds()"


def test_stage_ordering_stable():
    """Stage ordering matters for pipeline gate checks. Pin it down."""
    expected = ["source", "trace", "filter", "chatml", "balance", "validate", "split"]
    assert [s.value for s in Stage.ordered()] == expected
