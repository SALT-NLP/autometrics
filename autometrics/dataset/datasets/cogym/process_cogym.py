# -*- coding: utf-8 -*-
"""
process_cogym.py — v2.3 (2025‑05‑13)
=====================================
*Single‑source of truth* for turning a CoGym JSON session trace into:

  • metadata columns
  • raw event dump (for debugging)
  • clean human‑readable transcript
  • best‑guess final outcome

---------------------------------------------------------------------------
WHY V2 .3?
---------------------------------------------------------------------------
Previous rewrites fixed many ordering bugs but two edge‑cases remained:

1. **Stale chat echoes.**  Every event carries the entire
   `current_chat_history`; replaying the last element produced duplicate
   `[USER]`/`[ASSISTANT]` lines.

2. **Inline messages skipped.**  When an event *also* had
   `current_chat_history`, the inline `SEND_TEAMMATE_MESSAGE(...)` text was
   ignored, so real chat turns disappeared.

This version enforces **one simple invariant** (see above) that resolves both.

---------------------------------------------------------------------------
MODIFYING THIS FILE?  READ THIS FIRST  ➜  🔒  INVARIANTS
---------------------------------------------------------------------------
1. Print the inline message from `SEND_TEAMMATE_MESSAGE` **always**.
2. For the cached history, print the last message **only if**
   `msg.timestamp == event.timestamp`.
3. Use `_dedup_append` at every insertion point; never build your own
   de‑dup logic.
4. Keep the main loop sorted by `event.timestamp`.

Follow those four rules and the transcript stays correct ✔.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ------------------------------------------------------------------ #
#  Regex helpers
# ------------------------------------------------------------------ #
_MESSAGE_RE = re.compile(r"message=(?:'|\")?(.*?)(?:'|\")?\)?$")
_PENDING_EDITOR_RE = re.compile(
    r"pending_action=EDITOR_UPDATE\(\s*text=(?:'|\")(.*?)(?:'|\")\s*\)\s*\)$",
    re.DOTALL,
)

# Tool prefixes we care about; keep in sync with the CoGym backend.
_TOOL_PREFIXES = (
    "INTERNET_SEARCH",
    "BUSINESS_SEARCH",
    "DISTANCE_MATRIX",
    "EXECUTE_JUPYTER_CELL",
    "REQUEST_TEAMMATE_CONFIRM",
    "ACCEPT_CONFIRMATION",
    "REJECT_CONFIRMATION",
    "EDITOR_UPDATE",
)

# ------------------------------------------------------------------ #
#  Tiny utilities
# ------------------------------------------------------------------ #
def _extract_message_from_action(action: str) -> str:
    """Return the inline `message=` payload (if any) from a SEND_TEAMMATE_MESSAGE."""
    if action.startswith("SEND_TEAMMATE_MESSAGE") and "message=" in action:
        return action.split("message=", 1)[1].rstrip(")").strip("\"' ")
    return ""


def _dedup_append(lines: List[str], new_line: str) -> None:
    """Append *new_line* unless it is identical to the previous line."""
    if new_line and (not lines or lines[-1] != new_line):
        lines.append(new_line)


# ------------------------------------------------------------------ #
#  Chat emission helper
# ------------------------------------------------------------------ #
def _maybe_append_chat(
    *,
    lines: List[str],
    chat_block: Optional[List[Dict[str, Any]]],
    outer_role: str,
    ev_ts: Optional[str],
) -> None:
    """
    Emit the final chat message from *chat_block* **only if** that message
    belongs to this event (timestamps match).  Skip cached echoes.
    """
    if not chat_block:
        return

    last_msg = chat_block[-1]
    chat_ts = last_msg.get("timestamp")
    if chat_ts and ev_ts and chat_ts != ev_ts:
        return  # stale echo from earlier event

    chat_txt = last_msg.get("message", "").strip()
    if not chat_txt:
        return

    chat_role = last_msg.get("role", "")
    if chat_role.startswith("user"):
        tag = "[USER]"
    elif chat_role in {"assistant", "agent"}:
        tag = "[ASSISTANT]"
    else:  # fallback to outer event role
        tag = "[USER]" if outer_role.startswith("user") else "[ASSISTANT]"

    _dedup_append(lines, f"{tag} {chat_txt}")


# ------------------------------------------------------------------ #
#  Public helpers
# ------------------------------------------------------------------ #
def parse_metadata(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Pull fixed metadata columns from the session root."""
    keys = [
        "userId",
        "sessionId",
        "modelName",
        "task",
        "query",
        "createdAt",
        "finishedAt",
        "agentRating",
        "communicationRating",
        "outcomeRating",
        "agentFeedback",
        "finished",
        "bookmarked",
        "agentType",
    ]
    return {k: trace.get(k) for k in keys}


def parse_conversation(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a raw per‑event dump (for debugging)."""
    history: List[Dict[str, Any]] = []

    for ev in sorted(trace.get("event_log", []), key=lambda e: e.get("timestamp", "")):
        entry = {
            "timestamp": ev.get("timestamp"),
            "role": ev.get("role"),
            "action": ev.get("action"),
        }

        msg = _extract_message_from_action(ev.get("action", ""))
        if not msg and ev.get("current_chat_history"):
            msg = ev["current_chat_history"][-1].get("message", "")
        if msg:
            entry["message"] = msg

        if "current_observation" in ev:
            obs = ev["current_observation"].get("public") or ev["current_observation"]
            if obs:
                entry["observation"] = obs

        history.append(entry)

    return history


def parse_formatted_conversation(trace: Dict[str, Any], *, strict: bool = False) -> str:
    """
    Convert the event log into a readable transcript that:
      • shows each chat line exactly once
      • keeps tool calls in correct relative order
    """
    lines: List[str] = []
    last_editor_text = ""

    # Seed with the top‑level query (if present)
    if (q := trace.get("query", "").strip()):
        _dedup_append(lines, f"[USER] {q}")

    # --------------------------- per‑event FSM --------------------------- #
    def _process_event(ev: Dict[str, Any]) -> None:
        nonlocal last_editor_text

        outer_role = ev.get("role", "")
        action: str = ev.get("action", "") or ""
        chat_block = ev.get("current_chat_history")
        ev_ts = ev.get("timestamp")

        # 1️⃣ Environment notices
        if outer_role == "environment":
            _dedup_append(lines, f"[ENVIRONMENT] {action.split('(', 1)[0]}")
            return

        # 2️⃣ Tool / editor / confirmation handling
        action_emitted = False
        if outer_role == "agent" and any(action.startswith(p) for p in _TOOL_PREFIXES):
            if action.startswith("REQUEST_TEAMMATE_CONFIRM") and "pending_action=EDITOR_UPDATE(" in action:
                m = _PENDING_EDITOR_RE.search(action)
                pending_text = m.group(1) if m else ""
                if pending_text != last_editor_text:
                    name, args = action.split("(", 1)
                    _dedup_append(lines, f"<FUNCTION_CALL {name} {args.rstrip(')')}>")
                    action_emitted = True
            else:
                if action.startswith("EDITOR_UPDATE"):
                    try:
                        txt = action.split("text=", 1)[1]
                        last_editor_text = txt.rstrip(")").lstrip("\"' ")
                    except Exception:
                        last_editor_text = ""
                name, args = action.split("(", 1)
                _dedup_append(lines, f"<FUNCTION_CALL {name} {args.rstrip(')')}>")
                action_emitted = True

        # 3️⃣ Inline SEND_TEAMMATE_MESSAGE (real chat turn)
        if action.startswith("SEND_TEAMMATE_MESSAGE"):
            if (msg_txt := _extract_message_from_action(action)):
                tag = "[USER]" if outer_role.startswith("user") else "[ASSISTANT]"
                _dedup_append(lines, f"{tag} {msg_txt}")

        # 4️⃣ Chat (cached) — only if it belongs to this event
        _maybe_append_chat(
            lines=lines,
            chat_block=chat_block,
            outer_role=outer_role,
            ev_ts=ev_ts,
        )

        # 5️⃣ Strict mode sanity
        if strict and not (
            action_emitted
            or action.startswith("SEND_TEAMMATE_MESSAGE")
            or chat_block
        ):
            raise ValueError(f"Unprocessed event: {action[:80]}")

    # --------------------------- main loop ------------------------------ #
    for ev in sorted(trace.get("event_log", []), key=lambda e: e.get("timestamp", "")):
        _process_event(ev)

    return "\n\n".join(lines)


# ------------------------------------------------------------------ #
#  Outcome extraction (legacy code, unchanged)
# ------------------------------------------------------------------ #
def parse_outcome(trace: Dict[str, Any]) -> str:  # noqa: C901
    """Best‑effort recovery of the task outcome (editor text, etc.)."""
    for ev in trace.get("event_log", []):
        if ev.get("action", "").startswith("FINISH"):
            pub = ev.get("current_observation", {}).get("public", {})
            for fld in ("result_editor", "travel_plan_editor", "lesson_plan_editor"):
                if pub.get(fld):
                    return pub[fld]

    for ev in trace.get("event_log", []):
        if ev.get("action", "").startswith(
            ("ACCEPT_CONFIRMATION", "PUT_AGENT_ASLEEP", "WAKE_AGENT_UP")
        ):
            pub = ev.get("current_observation", {}).get("public", {})
            for fld in ("result_editor", "travel_plan_editor", "lesson_plan_editor"):
                if pub.get(fld):
                    return pub[fld]

    for ev in reversed(trace.get("event_log", [])):
        if ev.get("action", "").startswith("EDITOR_UPDATE"):
            pub = ev.get("current_observation", {}).get("public", {})
            for fld in ("result_editor", "travel_plan_editor", "lesson_plan_editor"):
                if pub.get(fld):
                    return pub[fld]

    for ev in reversed(trace.get("event_log", [])):
        act = ev.get("action", "")
        if act.startswith("REQUEST_TEAMMATE_CONFIRM") and "pending_action=EDITOR_UPDATE" in act:
            m = _PENDING_EDITOR_RE.search(act)
            if m:
                return m.group(1)

    for ev in reversed(trace.get("event_log", [])):
        if ev.get("action", "").startswith("SEND_TEAMMATE_MESSAGE"):
            if (m := _MESSAGE_RE.search(ev["action"])):
                return m.group(1)

    return ""


# ------------------------------------------------------------------ #
#  Batch helper & CLI
# ------------------------------------------------------------------ #
def process_directory(input_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fp in sorted(input_dir.glob("*.json")):
        trace = json.loads(fp.read_text(encoding="utf-8"))
        row = parse_metadata(trace)
        row["conversation"] = json.dumps(parse_conversation(trace), ensure_ascii=False)
        row["formatted_conversation"] = parse_formatted_conversation(trace)
        row["outcome"] = parse_outcome(trace)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[df["outcome"].str.strip().astype(bool)]   # keep rows where outcome ≠ ""
    return df


def _cli() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Convert CoGym JSON traces → CSV")
    p.add_argument("--input_dir", "-i", type=Path, required=True, help="Dir with *.json traces")
    p.add_argument("--output_csv", "-o", type=Path, required=True, help="Destination CSV file")
    args = p.parse_args()

    df = process_directory(args.input_dir)
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved {len(df):,} record(s) → {args.output_csv.resolve()}")


if __name__ == "__main__":
    _cli()