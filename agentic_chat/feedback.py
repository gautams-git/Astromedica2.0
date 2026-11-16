"""Feedback sink used by POST /feedback. The agent never reads this file."""
import json
import os
import re
import threading
import time

# Outside patients/: the app lists that folder's sub-folders as charts.
DEFAULT_PATH = "data/chat_feedback.jsonl"
MAX_COMMENT = 500                  # characters kept of the comment
MAX_SESSION_ID = 100
MAX_TURN = 100
_SIGNAL = re.compile(r"^[a-z0-9_-]{1,32}$")
_LOCK = threading.Lock()


class FeedbackInvalid(ValueError):
    """The request cannot be accepted; the message says why."""


class FeedbackFull(RuntimeError):
    """The feedback file reached its size limit."""


def _path() -> str:
    return os.getenv("RL_FEEDBACK_PATH", DEFAULT_PATH)


def _max_bytes() -> int:
    return int(os.getenv("RL_FEEDBACK_MAX_BYTES", "5000000"))


def record_feedback(user_id: str, session_id: str, turn_index: int,
                    signal: str, comment: str = "") -> None:
    if not user_id:
        raise FeedbackInvalid("no user")
    session_id = str(session_id or "").strip()
    if not session_id or len(session_id) > MAX_SESSION_ID:
        raise FeedbackInvalid(f"session_id must be 1 to {MAX_SESSION_ID} characters")
    if isinstance(turn_index, bool):
        raise FeedbackInvalid("turn_index must be an integer")
    try:
        turn_index = int(turn_index)
    except (TypeError, ValueError):
        raise FeedbackInvalid("turn_index must be an integer")
    if not 0 <= turn_index <= MAX_TURN:
        raise FeedbackInvalid(f"turn_index must be between 0 and {MAX_TURN}")
    signal = str(signal or "").strip().lower()
    if not _SIGNAL.match(signal):
        raise FeedbackInvalid("signal must be 1 to 32 characters: a-z, 0-9, _ or -")
    comment = str(comment or "").strip()[:MAX_COMMENT]

    line = json.dumps({
        "ts": time.time(),
        "user_id": user_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "signal": signal,
        "comment": comment,
    }, ensure_ascii=False) + "\n"

    path = _path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _LOCK:
        if os.path.exists(path) and os.path.getsize(path) + len(line.encode("utf-8")) > _max_bytes():
            raise FeedbackFull("the feedback file is full")
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
