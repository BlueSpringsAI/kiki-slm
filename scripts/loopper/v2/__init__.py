"""v2 dataset construction pipeline.

A per-intent, parallel, resumable, instrumented pipeline for building the
Loopper SLM training dataset. See docs/v2-pipeline-design.md for the full
architecture and docs/v2-pipeline-runbook.md for operational use.

The three load-bearing primitives:

    error_kinds.py  — taxonomy of every failure mode the pipeline can produce
    ledger.py       — SQLite-backed source of truth for record state
    validators.py   — pure validators, one per error_kind

All stage scripts, fixers, and the dashboard read from / write to the ledger;
all of them emit error_kinds from the taxonomy; all of them call validators
to produce those error_kinds. There is no other path for state to flow.
"""
