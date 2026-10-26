# Room Agent Chat Backend

Phase 05 adds a dependency-free Python chat pipeline. The backend owns normalization,
filtering, duplicate handling, priority scoring, intent extraction, short-lived
conversation context and LLM response validation. It does not execute Unity actions.

## Run tests

From the repository root:

```powershell
C:\Users\qviet\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s backend/tests
```

## Run the local API

```powershell
C:\Users\qviet\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m backend.api.chat
```

The mock provider is used by this command, so the API is testable without network access.

## Endpoints

- `POST /chat/message` — accepts `viewer_id` and `text`, normalizes, filters and queues the message.
- `GET /chat/queue` — returns the bounded attention queue.
- `GET /chat/state` — returns queue size, context size and the latest structured response.

Future systems call `ChatService.process_next()` to dequeue one message at a time.
This deliberately limits LLM work and prevents a chat burst from causing simultaneous calls.

## Operator-controlled runtime

The backend starts only when an operator explicitly runs
`AgentScripts/phase23/start_runtime_phase23.ps1 -Action start`. There is no
Scheduled Task, watchdog, automatic restart, or crash-recovery loop. After a
crash it remains stopped until that command is run again.

Use the same script with `-Action stop` for cooperative shutdown. See
`AgentScripts/phase23/README.md` for health, lock, and stale-lock checks.
