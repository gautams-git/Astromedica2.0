# Agentic chat and RL

The live chat switch is the existing `agentic_chat` package: `CHAT_MODE=classic` (default) or `agentic`, with `RL_HINT` and `RL_HINT_SHADOW`. This change does not replace that controller. It stores the session, turns, and RL output so more than one Gunicorn worker can resume the same chat. `Agentic_Chat_Doc.html` was not in this workspace.

## What already runs

`jataka_logic.run_logic` calls the hitlist and organ agents and, when `AGENTIC_DISEASE_LOGIC` and `AGENTIC_COMBINED` are on, the rashi, disease-first, disease-compare, and combined wrappers. Those wrappers call the existing engines. Their scoring was not edited.

The patient chat is `chat_module.py`, coordinated by `orchestrator.py`. The orchestrator prepends a fixed safety constraint. Finish detection, cluster scores, and the question bank stay in `chat_module.py`.

## Flags

| Variable | Default | Effect |
| --- | --- | --- |
| `CHAT_MODE` | `classic` | Stored on the session. The patient reply still comes from `chat_module`. |
| `RL_HINT_MODE` | `off` | `shadow` or `enabled` records `rl_hint_status=recorded_not_applied`. The hint does not change the reply or the finish decision. |
| `MODEL_PROVIDER` | `vertex` | `vertex` keeps Application Default Credentials. `lfm3` is selectable and is not wired into the live reply. |
| `RL_CONFIG_STRICT` | false | When true, an RL table reload that cannot read Firestore does not substitute bootstrap data. |

`rl_failure_blocks_reply()` is false. RL exceptions in the chat path are printed and the reply still returns.

## Durable state

Every saved turn is still written to `patients/{id}/chat_log.json`. When a request has an authenticated user, the turn is also appended under `chats/{sessionId}/turns`. Session fields needed to resume are stored on the chat document. A worker that does not have the session calls `recover_session` and continues only if the authenticated user owns it.

RL output is still `patients/{id}/RL Output.json`. After that file is written, the same payload is stored under `chats/{sessionId}/rl_outputs` for each account that owns the patient. The results page still reads the JSON file.

## RL data versus shared knowledge

Symptom libraries, aliases, branch hints, stop words, and organ families load through `rl_enhancer.config_loader` as they do today. The migration copies the bootstrap tables into `rl_config`. Training must not write those records into `knowledge_base`. A change to shared domain data is a reviewed migration with a checksum, not an automatic RL write.
