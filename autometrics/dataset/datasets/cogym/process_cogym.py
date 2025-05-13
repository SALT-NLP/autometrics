import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

####################################################################
# Utility helpers
####################################################################

_MESSAGE_RE = re.compile(r"message=(?:'|\")?(.*?)(?:'|\")?\)?$")


def _extract_message_from_action(action: str) -> str:
    """Return the text inside `message=` for a SEND_TEAMMATE_MESSAGE action."""
    if action and action.startswith("SEND_TEAMMATE_MESSAGE") and "message=" in action:
        raw = action.split("message=", 1)[1]
        if raw.endswith(")"):
            raw = raw[:-1]
        return raw.strip("\"' ")
    return ""


def _dedup_append(lines: List[str], new_line: str) -> None:
    """Append *new_line* unless it is identical to the previous line."""
    if new_line and (not lines or lines[-1] != new_line):
        lines.append(new_line)

####################################################################
# Metadata helpers (unchanged)
####################################################################

def parse_metadata(trace: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "userId", "sessionId", "modelName", "task", "query",
        "createdAt", "finishedAt", "agentRating",
        "communicationRating", "outcomeRating", "agentFeedback",
        "finished", "bookmarked", "agentType",
    ]
    return {k: trace.get(k) for k in keys}

####################################################################
# Raw JSON‑style conversation (unchanged)
####################################################################

def parse_conversation(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for ev in trace.get("event_log", []):
        entry = {
            "timestamp": ev.get("timestamp"),
            "role": ev.get("role"),
            "action": ev.get("action"),
        }

        # Friendly text if available
        msg = _extract_message_from_action(ev.get("action", ""))
        if not msg and ev.get("current_chat_history"):
            msg = ev["current_chat_history"][-1].get("message", "")
        if msg:
            entry["message"] = msg

        # Expose public observations (handy for debugging)
        if "current_observation" in ev:
            obs = ev["current_observation"].get("public") or ev["current_observation"]
            if obs:
                entry["observation"] = obs

        history.append(entry)
    return history

####################################################################
# ***  ENHANCED *** conversation formatter
####################################################################

def _is_redundant_confirm(action: str) -> bool:
    """
    True if the action is a REQUEST_TEAMMATE_CONFIRM whose pending_action is
    literally an EDITOR_UPDATE(...).  In those cases we suppress it because
    the matching EDITOR_UPDATE event carries the same payload.
    """
    if not action.startswith("REQUEST_TEAMMATE_CONFIRM"):
        return False
    # quick & dirty, but robust enough: look for 'pending_action=EDITOR_UPDATE('
    return "pending_action=EDITOR_UPDATE(" in action


def parse_formatted_conversation(trace: Dict[str, Any]) -> str:
    """
    Produce a clean, chronological transcript with tags:

      [USER] …           real user messages
      [ASSISTANT] …      assistant replies
      [ENVIRONMENT] …    environment events (START, etc.)
      <FUNCTION_CALL …>  tool / editor invocations

    Each block is separated by a **blank line** for easier reading.
    """

    lines: List[str] = []

    # ------------------------------------------------------------------
    # 0) Always start with the initial user query if we have it
    # ------------------------------------------------------------------
    initial_query = (trace.get("query") or "").strip()
    if initial_query:
        _dedup_append(lines, f"[USER] {initial_query}")

    evs = trace.get("event_log", [])
    if not evs:
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # 1) Walk the event log in order
    # ------------------------------------------------------------------
    for ev in evs:
        outer_role = ev.get("role", "")
        action     = ev.get("action", "") or ""

        # ---- ENVIRONMENT ------------------------------------------------
        if outer_role == "environment":
            _dedup_append(lines, f"[ENVIRONMENT] {action.split('(', 1)[0]}")
            continue

        # ---- USER messages via SEND_TEAMMATE_MESSAGE --------------------
        if outer_role.startswith("user_") and action.startswith("SEND_TEAMMATE_MESSAGE"):
            msg_txt = _extract_message_from_action(action)
            if msg_txt:
                _dedup_append(lines, f"[USER] {msg_txt}")
            continue

        # ---- ASSISTANT chat via SEND_TEAMMATE_MESSAGE -------------------
        if outer_role == "agent" and action.startswith("SEND_TEAMMATE_MESSAGE"):
            msg_txt = _extract_message_from_action(action)
            if msg_txt:
                _dedup_append(lines, f"[ASSISTANT] {msg_txt}")
            continue

        # ---- CHAT EVENTS pulled from current_chat_history ---------------
        if ev.get("current_chat_history"):
            msg_obj   = ev["current_chat_history"][-1]
            chat_role = msg_obj.get("role", "")
            chat_txt  = msg_obj.get("message", "").strip()

            if chat_role == "user" and chat_txt:
                _dedup_append(lines, f"[USER] {chat_txt}")
                continue
            if chat_role in ("assistant", "agent") and chat_txt:
                _dedup_append(lines, f"[ASSISTANT] {chat_txt}")
                continue

        # ---- TOOL / FUNCTION calls by assistant -------------------------
        if outer_role == "agent":
            # Skip redundant confirmation events that merely echo an EDITOR_UPDATE
            if _is_redundant_confirm(action):
                continue

            if any(action.startswith(p) for p in (
                "INTERNET_SEARCH", "BUSINESS_SEARCH", "DISTANCE_MATRIX",
                "EXECUTE_JUPYTER_CELL", "REQUEST_TEAMMATE_CONFIRM",
                "ACCEPT_CONFIRMATION", "EDITOR_UPDATE",
            )):
                name, args = action.split("(", 1)
                _dedup_append(lines, f"<FUNCTION_CALL {name} {args.rstrip(')')}>")
                continue

        # ---- Everything else : ignore -----------------------------------

    # Join with a blank line between entries
    return "\n\n".join(lines)

####################################################################
# Outcome extraction (unchanged)
####################################################################

def parse_outcome(trace: Dict[str, Any]) -> str:
    for ev in trace.get("event_log", []):
        if ev.get("action", "").startswith("FINISH"):
            pub = ev.get("current_observation", {}).get("public", {})
            for fld in ("result_editor", "travel_plan_editor", "lesson_plan_editor"):
                if pub.get(fld):
                    return pub[fld]
    for ev in trace.get("event_log", []):
        if ev.get("action", "").startswith("ACCEPT_CONFIRMATION"):
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
        if ev.get("action", "").startswith("SEND_TEAMMATE_MESSAGE"):
            m = re.search(r"message=(\"|')(.*?)(\1)", ev["action"])
            if m:
                return m.group(2)
    return ""

####################################################################
# Directory‑level batch helper (unchanged)
####################################################################

def process_directory(input_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fp in input_dir.glob("*.json"):
        with fp.open("r", encoding="utf-8") as f:
            trace = json.load(f)

        meta   = parse_metadata(trace)
        convo  = parse_conversation(trace)
        fmt    = parse_formatted_conversation(trace)
        outcome= parse_outcome(trace)

        row = meta.copy()
        row["conversation"]           = json.dumps(convo, ensure_ascii=False)
        row["formatted_conversation"] = fmt
        row["outcome"]                = outcome
        rows.append(row)

    return pd.DataFrame(rows)

####################################################################
# CLI entry point (unchanged)
####################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract session traces into a CSV file for analysis."
    )
    parser.add_argument(
        "--input_dir", "-i", type=Path, required=True,
        help="Directory containing *.json session traces"
    )
    parser.add_argument(
        "--output_csv", "-o", type=Path, required=True,
        help="Location where the CSV should be written"
    )
    args = parser.parse_args()

    df = process_directory(args.input_dir)
    df.to_csv(args.output_csv, index=False)
    print(f"✅  Saved {len(df):,} record(s) → {args.output_csv.resolve()}")

if __name__ == "__main__":
    main()

# Example usage:
# python process_cogym.py --input_dir /path/to/input --output_csv /path/to/output.csv
