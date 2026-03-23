#!/usr/bin/env python3
"""Comprehensive analysis of raw Freshdesk ticket data for Kiki SLM dataset planning.

Produces a detailed report covering:
- Volume & language distribution
- Conversation depth (single-turn vs multi-turn)
- Priority/urgency distribution
- Status distribution (resolved vs open)
- Message length statistics
- Tag/category analysis
- Sentiment distribution
- Time patterns
- Data quality assessment
- Dataset creation recommendations

Usage:
    python scripts/analyze_raw_tickets.py --input-dir raw_tickets/
    python scripts/analyze_raw_tickets.py --input-dir raw_tickets/ --output-report reports/ticket_analysis.md
    python scripts/analyze_raw_tickets.py --input-dir raw_tickets/ --sample 10000  # analyze subset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def load_tickets(input_dir: str, sample: int | None = None) -> list[dict]:
    """Load all ticket JSON files from directory."""
    ticket_dir = Path(input_dir)
    files = sorted(ticket_dir.glob("*.json"))

    if not files:
        print(f"ERROR: No JSON files found in {input_dir}")
        sys.exit(1)

    if sample and sample < len(files):
        import random
        random.seed(42)
        files = random.sample(files, sample)

    tickets = []
    errors = 0
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            tickets.append(data)
        except (json.JSONDecodeError, Exception):
            errors += 1

    print(f"Loaded {len(tickets):,} tickets ({errors} errors) from {input_dir}")
    return tickets


def analyze(tickets: list[dict]) -> dict:
    """Run all analyses and return structured results."""
    results = {}
    total = len(tickets)

    # ================================================================
    # 1. BASIC STATS
    # ================================================================
    results["total_tickets"] = total

    # ================================================================
    # 2. LANGUAGE DISTRIBUTION
    # ================================================================
    languages = Counter()
    for t in tickets:
        lang = t.get("ticket", {}).get("detected_language", "unknown")
        languages[lang] += 1
    results["languages"] = dict(languages.most_common())

    # ================================================================
    # 3. CONVERSATION DEPTH
    # ================================================================
    conv_depths = Counter()
    has_agent_response = 0
    has_customer_followup = 0
    multi_turn_tickets = 0
    total_conversations = 0

    for t in tickets:
        convs = t.get("conversations", [])
        n = len(convs)
        conv_depths[n] += 1
        total_conversations += n

        if n > 0:
            has_incoming = any(c.get("incoming", False) for c in convs)
            has_outgoing = any(not c.get("incoming", True) for c in convs)
            if has_outgoing:
                has_agent_response += 1
            if has_incoming and n > 0:
                has_customer_followup += 1
            if n >= 2:
                multi_turn_tickets += 1

    results["conversation_depth"] = {
        "distribution": dict(sorted(conv_depths.items())[:20]),
        "zero_conversations": conv_depths.get(0, 0),
        "single_conversation": conv_depths.get(1, 0),
        "multi_turn_2_plus": sum(v for k, v in conv_depths.items() if k >= 2),
        "multi_turn_5_plus": sum(v for k, v in conv_depths.items() if k >= 5),
        "multi_turn_10_plus": sum(v for k, v in conv_depths.items() if k >= 10),
        "max_depth": max(conv_depths.keys()) if conv_depths else 0,
        "avg_depth": total_conversations / total if total else 0,
        "has_agent_response": has_agent_response,
        "has_customer_followup": has_customer_followup,
    }

    # ================================================================
    # 4. PRIORITY DISTRIBUTION
    # ================================================================
    priorities = Counter()
    for t in tickets:
        pri = t.get("ticket", {}).get("priority", "unknown")
        priorities[pri] += 1
    results["priorities"] = dict(priorities.most_common())
    # Freshdesk: 1=low, 2=medium, 3=high, 4=urgent

    # ================================================================
    # 5. STATUS DISTRIBUTION
    # ================================================================
    statuses = Counter()
    for t in tickets:
        status = t.get("ticket", {}).get("status", "unknown")
        statuses[status] += 1
    results["statuses"] = dict(statuses.most_common())
    # Freshdesk: 2=open, 3=pending, 4=resolved, 5=closed

    # ================================================================
    # 6. SOURCE DISTRIBUTION
    # ================================================================
    sources = Counter()
    for t in tickets:
        src = t.get("ticket", {}).get("source", "unknown")
        sources[src] += 1
    results["sources"] = dict(sources.most_common())
    # Freshdesk: 1=email, 2=portal, 3=phone, 7=chat, 9=feedback, 10=outbound

    # ================================================================
    # 7. MESSAGE LENGTH STATISTICS
    # ================================================================
    desc_lengths = []
    conv_lengths = []
    empty_descriptions = 0

    for t in tickets:
        desc = t.get("ticket", {}).get("description_text", "") or ""
        if len(desc.strip()) < 5:
            empty_descriptions += 1
        else:
            desc_lengths.append(len(desc))

        for c in t.get("conversations", []):
            body = c.get("body_text", "") or ""
            if body.strip():
                conv_lengths.append(len(body))

    results["message_lengths"] = {
        "description": {
            "count": len(desc_lengths),
            "empty": empty_descriptions,
            "min": min(desc_lengths) if desc_lengths else 0,
            "max": max(desc_lengths) if desc_lengths else 0,
            "avg": sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0,
            "median": sorted(desc_lengths)[len(desc_lengths) // 2] if desc_lengths else 0,
            "under_50_chars": sum(1 for l in desc_lengths if l < 50),
            "50_to_500_chars": sum(1 for l in desc_lengths if 50 <= l < 500),
            "500_to_2000_chars": sum(1 for l in desc_lengths if 500 <= l < 2000),
            "over_2000_chars": sum(1 for l in desc_lengths if l >= 2000),
        },
        "conversations": {
            "count": len(conv_lengths),
            "avg": sum(conv_lengths) / len(conv_lengths) if conv_lengths else 0,
            "median": sorted(conv_lengths)[len(conv_lengths) // 2] if conv_lengths else 0,
        },
    }

    # ================================================================
    # 8. TAG ANALYSIS
    # ================================================================
    all_tags = Counter()
    tickets_with_tags = 0
    for t in tickets:
        tags = t.get("ticket", {}).get("tags", [])
        if tags:
            tickets_with_tags += 1
            for tag in tags:
                all_tags[tag] += 1

    results["tags"] = {
        "tickets_with_tags": tickets_with_tags,
        "tickets_without_tags": total - tickets_with_tags,
        "unique_tags": len(all_tags),
        "top_50_tags": dict(all_tags.most_common(50)),
    }

    # ================================================================
    # 9. SENTIMENT DISTRIBUTION
    # ================================================================
    sentiments = []
    for t in tickets:
        score = t.get("ticket", {}).get("sentiment_score")
        if score is not None:
            sentiments.append(score)

    if sentiments:
        results["sentiment"] = {
            "count": len(sentiments),
            "avg": sum(sentiments) / len(sentiments),
            "min": min(sentiments),
            "max": max(sentiments),
            "very_negative_0_20": sum(1 for s in sentiments if s <= 20),
            "negative_21_40": sum(1 for s in sentiments if 21 <= s <= 40),
            "neutral_41_60": sum(1 for s in sentiments if 41 <= s <= 60),
            "positive_61_80": sum(1 for s in sentiments if 61 <= s <= 80),
            "very_positive_81_100": sum(1 for s in sentiments if s >= 81),
        }

    # ================================================================
    # 10. TIME ANALYSIS
    # ================================================================
    months = Counter()
    days_of_week = Counter()
    hours = Counter()
    for t in tickets:
        created = t.get("ticket", {}).get("created_at", "")
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                months[dt.strftime("%Y-%m")] += 1
                days_of_week[dt.strftime("%A")] += 1
                hours[dt.hour] += 1
            except (ValueError, AttributeError):
                pass

    results["time"] = {
        "months": dict(sorted(months.items())),
        "days_of_week": dict(days_of_week.most_common()),
        "peak_hours": dict(sorted(hours.items())),
    }

    # ================================================================
    # 11. SUBJECT ANALYSIS
    # ================================================================
    subjects = Counter()
    subject_lengths = []
    for t in tickets:
        subj = t.get("ticket", {}).get("subject", "") or ""
        subject_lengths.append(len(subj))
        # Normalize for grouping
        norm = subj.strip().lower()[:50]
        if norm:
            subjects[norm] += 1

    results["subjects"] = {
        "unique_subjects": len(subjects),
        "avg_length": sum(subject_lengths) / len(subject_lengths) if subject_lengths else 0,
        "top_30_subjects": dict(subjects.most_common(30)),
        "repeated_subjects_5_plus": sum(1 for s, c in subjects.items() if c >= 5),
    }

    # ================================================================
    # 12. AUTOMATED vs HUMAN TICKET DETECTION
    # ================================================================
    automated_signals = 0
    for t in tickets:
        ticket = t.get("ticket", {})
        desc = (ticket.get("description_text", "") or "").lower()
        convs = t.get("conversations", [])
        subj = (ticket.get("subject", "") or "").lower()

        is_automated = False
        # Signals of automated tickets:
        if "automatisch generierte nachricht" in desc:
            is_automated = True
        elif "auto-generated" in desc or "do not reply" in desc:
            is_automated = True
        elif "noreply" in (ticket.get("support_email", "") or "").lower():
            is_automated = True
        elif len(convs) == 0 and ticket.get("sentiment_score", 50) == 50:
            # No conversations + neutral sentiment = likely automated
            is_automated = True
        elif "[name]" == subj.strip():
            is_automated = True

        if is_automated:
            automated_signals += 1

    results["automated_detection"] = {
        "likely_automated": automated_signals,
        "likely_human": total - automated_signals,
        "automated_percent": automated_signals / total * 100 if total else 0,
    }

    # ================================================================
    # 13. CUSTOM FIELDS
    # ================================================================
    custom_field_names = Counter()
    custom_field_filled = Counter()
    for t in tickets:
        cf = t.get("ticket", {}).get("custom_fields", {})
        if cf:
            for k, v in cf.items():
                custom_field_names[k] += 1
                if v is not None and str(v).strip():
                    custom_field_filled[k] += 1

    results["custom_fields"] = {
        "fields_found": dict(custom_field_names.most_common()),
        "fields_filled": dict(custom_field_filled.most_common()),
    }

    # ================================================================
    # 14. USABLE FOR TRAINING ASSESSMENT
    # ================================================================
    usable_single_turn = 0
    usable_multi_turn = 0
    usable_with_resolution = 0

    for t in tickets:
        ticket = t.get("ticket", {})
        desc = (ticket.get("description_text", "") or "").strip()
        convs = t.get("conversations", [])
        status = ticket.get("status", 0)
        spam = ticket.get("spam", False)

        if spam or len(desc) < 20:
            continue

        # Check if resolved/closed
        is_resolved = status in (4, 5)

        # Check for agent response
        has_agent = any(not c.get("incoming", True) for c in convs)
        agent_body = ""
        for c in convs:
            if not c.get("incoming", True):
                agent_body = (c.get("body_text", "") or "").strip()
                break

        if is_resolved and len(desc) >= 20:
            usable_single_turn += 1

        if is_resolved and has_agent and len(agent_body) >= 20:
            usable_with_resolution += 1

        if is_resolved and len(convs) >= 2:
            usable_multi_turn += 1

    results["training_usability"] = {
        "usable_single_turn": usable_single_turn,
        "usable_with_agent_response": usable_with_resolution,
        "usable_multi_turn": usable_multi_turn,
        "not_usable": total - usable_single_turn,
    }

    return results


def generate_report(results: dict, output_path: str | None = None) -> str:
    """Generate markdown report from analysis results."""
    total = results["total_tickets"]

    lines = []
    lines.append("# Kiki SLM — Raw Ticket Data Analysis Report")
    lines.append("")
    lines.append(f"> Analyzed **{total:,}** tickets")
    lines.append(f"> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ---- SUMMARY ----
    lines.append("---")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")

    tu = results["training_usability"]
    lines.append(f"| Metric | Count | % of Total |")
    lines.append(f"|:-------|------:|:-----------|")
    lines.append(f"| Total tickets | {total:,} | 100% |")
    lines.append(f"| Usable for single-turn training | {tu['usable_single_turn']:,} | {tu['usable_single_turn']/total*100:.1f}% |")
    lines.append(f"| Usable with agent response | {tu['usable_with_agent_response']:,} | {tu['usable_with_agent_response']/total*100:.1f}% |")
    lines.append(f"| Usable for multi-turn training | {tu['usable_multi_turn']:,} | {tu['usable_multi_turn']/total*100:.1f}% |")
    lines.append(f"| Likely automated (filter out) | {results['automated_detection']['likely_automated']:,} | {results['automated_detection']['automated_percent']:.1f}% |")
    lines.append("")

    # ---- LANGUAGE ----
    lines.append("---")
    lines.append("")
    lines.append("## Language Distribution")
    lines.append("")
    lines.append("| Language | Tickets | % |")
    lines.append("|:---------|--------:|:--|")
    for lang, count in sorted(results["languages"].items(), key=lambda x: -x[1]):
        lines.append(f"| {lang} | {count:,} | {count/total*100:.1f}% |")
    lines.append("")

    # ---- CONVERSATIONS ----
    lines.append("---")
    lines.append("")
    lines.append("## Conversation Depth")
    lines.append("")
    cd = results["conversation_depth"]
    lines.append(f"| Metric | Count | % |")
    lines.append(f"|:-------|------:|:--|")
    lines.append(f"| Zero conversations (no replies) | {cd['zero_conversations']:,} | {cd['zero_conversations']/total*100:.1f}% |")
    lines.append(f"| Single conversation | {cd['single_conversation']:,} | {cd['single_conversation']/total*100:.1f}% |")
    lines.append(f"| Multi-turn (2+ messages) | {cd['multi_turn_2_plus']:,} | {cd['multi_turn_2_plus']/total*100:.1f}% |")
    lines.append(f"| Deep threads (5+ messages) | {cd['multi_turn_5_plus']:,} | {cd['multi_turn_5_plus']/total*100:.1f}% |")
    lines.append(f"| Very deep (10+ messages) | {cd['multi_turn_10_plus']:,} | {cd['multi_turn_10_plus']/total*100:.1f}% |")
    lines.append(f"| Has agent response | {cd['has_agent_response']:,} | {cd['has_agent_response']/total*100:.1f}% |")
    lines.append(f"| Max conversation depth | {cd['max_depth']} | |")
    lines.append(f"| Average depth | {cd['avg_depth']:.1f} | |")
    lines.append("")
    lines.append("### Depth distribution (first 20)")
    lines.append("")
    lines.append("| # Conversations | Tickets |")
    lines.append("|:----------------|--------:|")
    for depth, count in sorted(results["conversation_depth"]["distribution"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {depth} | {count:,} |")
    lines.append("")

    # ---- PRIORITY ----
    lines.append("---")
    lines.append("")
    lines.append("## Priority Distribution")
    lines.append("")
    pri_names = {1: "Low", 2: "Medium", 3: "High", 4: "Urgent"}
    lines.append("| Priority | Tickets | % |")
    lines.append("|:---------|--------:|:--|")
    for pri, count in sorted(results["priorities"].items(), key=lambda x: -x[1]):
        name = pri_names.get(pri, f"Unknown ({pri})")
        lines.append(f"| {name} ({pri}) | {count:,} | {count/total*100:.1f}% |")
    lines.append("")

    # ---- STATUS ----
    lines.append("---")
    lines.append("")
    lines.append("## Status Distribution")
    lines.append("")
    status_names = {2: "Open", 3: "Pending", 4: "Resolved", 5: "Closed"}
    lines.append("| Status | Tickets | % |")
    lines.append("|:-------|--------:|:--|")
    for status, count in sorted(results["statuses"].items(), key=lambda x: -x[1]):
        name = status_names.get(status, f"Unknown ({status})")
        lines.append(f"| {name} ({status}) | {count:,} | {count/total*100:.1f}% |")
    lines.append("")

    # ---- SOURCE ----
    lines.append("---")
    lines.append("")
    lines.append("## Source Distribution")
    lines.append("")
    source_names = {1: "Email", 2: "Portal", 3: "Phone", 7: "Chat", 9: "Feedback Widget", 10: "Outbound Email"}
    lines.append("| Source | Tickets | % |")
    lines.append("|:-------|--------:|:--|")
    for src, count in sorted(results["sources"].items(), key=lambda x: -x[1]):
        name = source_names.get(src, f"Unknown ({src})")
        lines.append(f"| {name} ({src}) | {count:,} | {count/total*100:.1f}% |")
    lines.append("")

    # ---- MESSAGE LENGTHS ----
    lines.append("---")
    lines.append("")
    lines.append("## Message Length Statistics")
    lines.append("")
    ml = results["message_lengths"]["description"]
    lines.append("### Initial customer message (description)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|:-------|------:|")
    lines.append(f"| Total with content | {ml['count']:,} |")
    lines.append(f"| Empty/very short | {ml['empty']:,} |")
    lines.append(f"| Average length | {ml['avg']:.0f} chars |")
    lines.append(f"| Median length | {ml['median']:,} chars |")
    lines.append(f"| Max length | {ml['max']:,} chars |")
    lines.append(f"| Under 50 chars | {ml['under_50_chars']:,} |")
    lines.append(f"| 50-500 chars | {ml['50_to_500_chars']:,} |")
    lines.append(f"| 500-2000 chars | {ml['500_to_2000_chars']:,} |")
    lines.append(f"| Over 2000 chars | {ml['over_2000_chars']:,} |")
    lines.append("")
    cl = results["message_lengths"]["conversations"]
    lines.append("### Conversation messages")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|:-------|------:|")
    lines.append(f"| Total messages | {cl['count']:,} |")
    lines.append(f"| Average length | {cl['avg']:.0f} chars |")
    lines.append(f"| Median length | {cl['median']:,} chars |")
    lines.append("")

    # ---- TAGS ----
    lines.append("---")
    lines.append("")
    lines.append("## Tags / Categories")
    lines.append("")
    tg = results["tags"]
    lines.append(f"- Tickets with tags: **{tg['tickets_with_tags']:,}** ({tg['tickets_with_tags']/total*100:.1f}%)")
    lines.append(f"- Tickets without tags: **{tg['tickets_without_tags']:,}** ({tg['tickets_without_tags']/total*100:.1f}%)")
    lines.append(f"- Unique tags: **{tg['unique_tags']}**")
    lines.append(f"- Tags used 5+ times: **{tg.get('top_50_tags', {}).get('count', len([v for v in tg.get('top_50_tags', {}).values() if isinstance(v, int) and v >= 5]))}**")
    lines.append("")
    if tg["top_50_tags"]:
        lines.append("### Top 50 Tags")
        lines.append("")
        lines.append("| Tag | Count |")
        lines.append("|:----|------:|")
        for tag, count in sorted(tg["top_50_tags"].items(), key=lambda x: -x[1]):
            lines.append(f"| {tag} | {count:,} |")
        lines.append("")

    # ---- SENTIMENT ----
    if "sentiment" in results:
        lines.append("---")
        lines.append("")
        lines.append("## Sentiment Distribution")
        lines.append("")
        s = results["sentiment"]
        lines.append(f"| Sentiment Range | Tickets | % |")
        lines.append(f"|:----------------|--------:|:--|")
        lines.append(f"| Very negative (0-20) | {s['very_negative_0_20']:,} | {s['very_negative_0_20']/s['count']*100:.1f}% |")
        lines.append(f"| Negative (21-40) | {s['negative_21_40']:,} | {s['negative_21_40']/s['count']*100:.1f}% |")
        lines.append(f"| Neutral (41-60) | {s['neutral_41_60']:,} | {s['neutral_41_60']/s['count']*100:.1f}% |")
        lines.append(f"| Positive (61-80) | {s['positive_61_80']:,} | {s['positive_61_80']/s['count']*100:.1f}% |")
        lines.append(f"| Very positive (81-100) | {s['very_positive_81_100']:,} | {s['very_positive_81_100']/s['count']*100:.1f}% |")
        lines.append(f"| Average | {s['avg']:.1f} | |")
        lines.append("")

    # ---- SUBJECTS ----
    lines.append("---")
    lines.append("")
    lines.append("## Subject Analysis")
    lines.append("")
    su = results["subjects"]
    lines.append(f"- Unique subjects: **{su['unique_subjects']:,}**")
    lines.append(f"- Average subject length: **{su['avg_length']:.0f}** chars")
    lines.append(f"- Subjects repeated 5+ times: **{su['repeated_subjects_5_plus']}** (potential templates/auto-tickets)")
    lines.append("")
    if su["top_30_subjects"]:
        lines.append("### Top 30 Most Common Subjects")
        lines.append("")
        lines.append("| Subject | Count |")
        lines.append("|:--------|------:|")
        for subj, count in sorted(su["top_30_subjects"].items(), key=lambda x: -x[1]):
            lines.append(f"| {subj[:60]} | {count:,} |")
        lines.append("")

    # ---- TIME ----
    lines.append("---")
    lines.append("")
    lines.append("## Time Patterns")
    lines.append("")
    if results["time"]["months"]:
        lines.append("### Monthly Volume")
        lines.append("")
        lines.append("| Month | Tickets |")
        lines.append("|:------|--------:|")
        for month, count in sorted(results["time"]["months"].items()):
            lines.append(f"| {month} | {count:,} |")
        lines.append("")

    if results["time"]["days_of_week"]:
        lines.append("### Day of Week")
        lines.append("")
        lines.append("| Day | Tickets |")
        lines.append("|:----|--------:|")
        for day, count in results["time"]["days_of_week"].items():
            lines.append(f"| {day} | {count:,} |")
        lines.append("")

    # ---- CUSTOM FIELDS ----
    if results["custom_fields"]["fields_found"]:
        lines.append("---")
        lines.append("")
        lines.append("## Custom Fields")
        lines.append("")
        lines.append("| Field | Present | Filled |")
        lines.append("|:------|--------:|-------:|")
        for field, count in results["custom_fields"]["fields_found"].items():
            filled = results["custom_fields"]["fields_filled"].get(field, 0)
            lines.append(f"| {field} | {count:,} | {filled:,} |")
        lines.append("")

    # ---- AUTOMATED DETECTION ----
    lines.append("---")
    lines.append("")
    lines.append("## Automated Ticket Detection")
    lines.append("")
    ad = results["automated_detection"]
    lines.append(f"| Type | Count | % |")
    lines.append(f"|:-----|------:|:--|")
    lines.append(f"| Likely automated | {ad['likely_automated']:,} | {ad['automated_percent']:.1f}% |")
    lines.append(f"| Likely human | {ad['likely_human']:,} | {100 - ad['automated_percent']:.1f}% |")
    lines.append("")
    lines.append("Detection signals: auto-generated message text, noreply email, zero conversations + neutral sentiment, template subjects.")
    lines.append("")

    # ---- TRAINING USABILITY ----
    lines.append("---")
    lines.append("")
    lines.append("## Training Data Usability Assessment")
    lines.append("")
    tu = results["training_usability"]
    lines.append(f"| Category | Count | % | Use For |")
    lines.append(f"|:---------|------:|:--|:--------|")
    lines.append(f"| Usable single-turn | {tu['usable_single_turn']:,} | {tu['usable_single_turn']/total*100:.1f}% | Intent + urgency training |")
    lines.append(f"| Usable with agent response | {tu['usable_with_agent_response']:,} | {tu['usable_with_agent_response']/total*100:.1f}% | Full SFT (intent + response) |")
    lines.append(f"| Usable multi-turn | {tu['usable_multi_turn']:,} | {tu['usable_multi_turn']/total*100:.1f}% | Multi-turn conversation training |")
    lines.append(f"| Not usable | {tu['not_usable']:,} | {tu['not_usable']/total*100:.1f}% | Filter out (spam, empty, unresolved) |")
    lines.append("")

    # ---- RECOMMENDATIONS ----
    lines.append("---")
    lines.append("")
    lines.append("## Recommendations for Dataset Creation")
    lines.append("")
    lines.append("### Immediate actions")
    lines.append("")
    lines.append(f"1. **Filter automated tickets** — {ad['likely_automated']:,} ({ad['automated_percent']:.1f}%) are likely auto-generated. Remove these.")
    lines.append(f"2. **Use resolved tickets with agent responses** — {tu['usable_with_agent_response']:,} tickets have both a customer message and agent resolution. These are the highest quality for SFT.")
    lines.append(f"3. **Extract multi-turn conversations** — {tu['usable_multi_turn']:,} tickets have 2+ messages. These teach the model conversation context.")

    top_lang = max(results["languages"].items(), key=lambda x: x[1]) if results["languages"] else ("?", 0)
    lines.append(f"4. **Language consideration** — Primary language is **{top_lang[0]}** ({top_lang[1]/total*100:.1f}%). Decide if training should be multilingual or language-filtered.")
    lines.append("")
    lines.append("### Dataset sizing estimate")
    lines.append("")
    lines.append(f"- After filtering automated + empty + unresolved: **~{tu['usable_with_agent_response']:,}** usable tickets")
    lines.append(f"- Of those, **~{tu['usable_multi_turn']:,}** have multi-turn conversations")
    lines.append(f"- Recommended: Use all usable tickets + GPT-4o annotation for intent/workflow labels")
    lines.append("")
    lines.append("### Next steps")
    lines.append("")
    lines.append("1. Export full 400K tickets from Freshdesk")
    lines.append("2. Run this analysis on full dataset")
    lines.append("3. Build Freshdesk JSON → ChatML converter")
    lines.append("4. GPT-4o batch annotation for intent/urgency/workflow labels")
    lines.append("5. Mix with existing public training data (60/25/15 split)")
    lines.append("6. Retrain and evaluate")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(report)
        print(f"\nReport saved to {path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze raw Freshdesk ticket data for Kiki SLM")
    parser.add_argument("--input-dir", required=True, help="Directory containing ticket JSON files")
    parser.add_argument("--output-report", default="reports/ticket_analysis.md", help="Output markdown report path")
    parser.add_argument("--output-json", default="reports/ticket_analysis.json", help="Output raw JSON results")
    parser.add_argument("--sample", type=int, default=None, help="Analyze only N random tickets")
    args = parser.parse_args()

    tickets = load_tickets(args.input_dir, args.sample)
    results = analyze(tickets)

    # Save raw JSON
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Raw results saved to {json_path}")

    # Generate and save report
    report = generate_report(results, args.output_report)

    # Print summary to stdout
    print(f"\n{'='*60}")
    print(f"  TICKET ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total tickets:          {results['total_tickets']:,}")
    print(f"  Languages:              {len(results['languages'])}")
    print(f"  With agent response:    {results['training_usability']['usable_with_agent_response']:,}")
    print(f"  Multi-turn (2+):        {results['conversation_depth']['multi_turn_2_plus']:,}")
    print(f"  Likely automated:       {results['automated_detection']['likely_automated']:,}")
    print(f"  Usable for training:    {results['training_usability']['usable_with_agent_response']:,}")
    print(f"\n  Full report: {args.output_report}")
    print(f"  Raw data:    {args.output_json}")


if __name__ == "__main__":
    main()
