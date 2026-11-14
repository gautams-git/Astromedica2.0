"""Feedback sink used by POST /feedback. The agent never reads this file."""
import json
import os
import time

_FEEDBACK_PATH = os.getenv("RL_FEEDBACK_PATH", "patients/_feedback.jsonl")


def record_feedback(user_id: str, session_id: str, turn_index: int,
                    signal: str, comment: str = "") -> None:
    entry = {
        "ts": time.time(),
        "user_id": user_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "signal": signal,
        "comment": comment,
    }
    os.makedirs(os.path.dirname(_FEEDBACK_PATH), exist_ok=True)
    with open(_FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")