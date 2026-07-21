# Logistics AI System

A local-first logistics assistant: a chat UI backed by a FastAPI orchestrator
that combines RAG (company docs/policies) with MCP tool calls (live
shipment/inbound/outbound/inventory/returns lookups), running against either
a local Ollama model or the Claude API.

## How it fits together

```
                        ┌─────────────────────┐
   browser  ───────────▶│  app/static (chat UI) │
                        └──────────┬───────────┘
                                   │ POST /chat (SSE stream)
                                   ▼
                        ┌─────────────────────┐
                        │  FastAPI app (:8010)  │
                        │  app/orchestrator/    │
                        └──────┬─────────┬──────┘
                               │         │
              ┌────────────────┘         └────────────────┐
              ▼                                            ▼
   ┌─────────────────────┐                     ┌─────────────────────────┐
   │  RAG (app/rag/)       │                     │  MCP tool clients        │
   │  hybrid search over   │                     │  (app/mcp_clients/)      │
   │  Qdrant + BM25         │                     └──────────┬──────────────┘
   └──────────┬────────────┘                                │
              │                                              ▼
              ▼                                   5 standalone MCP servers
   ┌─────────────────────┐                        (mcp_servers/*, :7001-7005)
   │  Qdrant (embedded,     │                        in-memory sample data:
   │  ./qdrant_storage)     │                        shipments / inbound / outbound
   └─────────────────────┘                        / inventory / returns

   Chat model: Ollama (local, default) or Claude API — one config switch.
   Query/document embeddings: Ollama (nomic-embed-text) either way.
   Session history: Redis.
```

Every request first goes through **domain/tool routing**
(`app/rag/router.py`, `app/mcp_clients/tool_router.py`) so the model only
ever sees the RAG domains and MCP tools that are actually relevant to the
question — this keeps prompts small (faster, cheaper) and stops the model
from fabricating a tool call for an unrelated question.

## Does it re-embed the documentation on every prompt or every restart?

**No.** Two separate paths, and they don't overlap:

1. **Document embedding** happens only when you explicitly run
   `make ingest`. It hashes every file under `ingestion/data/`, skips
   anything unchanged since the last run (`ingestion/manifest.json`), and
   writes vectors into Qdrant's on-disk storage (`./qdrant_storage`). That
   storage persists across app restarts — starting the app never touches
   ingestion.
2. **Query-time embedding** happens once per chat question — only the
   user's question text (a few words) gets embedded, to search against the
   vectors that are already stored. That's a tiny embedding call per
   question, not the corpus.

So cost/latency from embeddings scales with how many questions get asked,
not with how much documentation exists or how many times you restart the
server. Re-embedding the whole corpus only happens if you explicitly run
`make ingest` again (and even then, only changed files are re-embedded) or
force it with `uv run python -m ingestion.cli rebuild --all`.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (manages the Python venv automatically)
- [Ollama](https://ollama.com) running locally, with these models pulled:
  ```
  ollama pull llama3.2          # chat model (if using the Ollama provider)
  ollama pull nomic-embed-text  # embeddings (always required, both providers)
  ```
- Redis running locally (`redis-server`) — used for session/conversation history
- A `.env` file (copy `.env.example` and fill in the blanks — see below)

## First-time setup

```sh
cp .env.example .env
make install       # uv sync
make ingest        # embed the sample docs into Qdrant (one-time; re-run whenever docs change)
```

## Running everything with one command

```sh
make up
```

This starts the 5 MCP servers in the background, waits until they're ready,
then runs the main app **in the foreground on port 8010** so you can watch
live logs. Open **http://localhost:8010** for the chat UI.

`Ctrl+C` stops the app and automatically tears down the background MCP
servers with it. If a terminal ever gets closed uncleanly and something is
left running, `make down` sweeps up any leftover MCP server processes.

Other useful targets:

| Command | What it does |
|---|---|
| `make up` | Start everything (MCP servers + app), one command, foreground app logs |
| `make down` | Stop any background MCP servers (safe to run any time, including after an unclean exit) |
| `make ingest` | Embed new/changed docs from `ingestion/data/` into Qdrant. **Stop the app first** — see note below |
| `make status` | Show what's been ingested per domain, and what's pending |
| `make test` | Run the unit test suite |
| `make eval` | Run the RAG retrieval eval (`tests/rag_eval/`) — hit@5 / MRR against real fixtures |

### Why `make ingest` and `make up` can't run at the same time

Qdrant is running in **embedded** mode (`QDRANT_URL=./qdrant_storage` in
`.env` — a local on-disk path, not a server URL). Embedded mode only allows
one process to hold that storage folder at a time. Stop the app
(`Ctrl+C` on `make up`) before running `make ingest`, and vice versa. If you
want to avoid this entirely, point `QDRANT_URL` at a real Qdrant server
instead (e.g. `http://localhost:6333`) — the codebase supports both
(`app/rag/vector_store.py` picks embedded vs. server mode based on whether
the URL looks like `http(s)://`).

## Switching between Ollama and Claude

Both live behind the same `ModelProvider` interface
(`app/model_providers/`) — swapping providers is a config change only, no
code changes. In `.env`:

```sh
# Local, free, private, but slow on CPU-only hardware:
CHAT_MODEL_PROVIDER=ollama
OLLAMA_CHAT_MODEL=llama3.2

# Claude API — faster and higher quality, costs API credits:
CHAT_MODEL_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-5
```

`EMBEDDING_PROVIDER` stays `ollama` either way — embeddings always run
locally via `nomic-embed-text`, regardless of which chat model you pick.

## Directory guide

| Path | What it is |
|---|---|
| `app/main.py` | FastAPI entrypoint; mounts `/chat`, `/health`, and the static chat UI |
| `app/orchestrator/` | Ties RAG + MCP tools + the model provider together per request |
| `app/rag/` | RAG plugin system — one `RagSource` per knowledge domain (compliance, support-docs, known-issues, inbound, outbound, inventory, returns) |
| `app/mcp_clients/` | HTTP clients + routing for the 5 MCP tool servers |
| `app/model_providers/` | `OllamaProvider` / `ClaudeProvider` — swappable chat backends |
| `app/auth/permissions.py` | Role → allowed RAG domains / MCP tools (`customer`, `support_agent`, `manager`) |
| `app/static/` | The chat UI (single self-contained HTML/JS page, no build step) |
| `mcp_servers/*` | 5 standalone FastAPI apps simulating shipments/inbound/outbound/inventory/returns systems (in-memory sample data) |
| `ingestion/` | CLI (`cli.py`) + pipeline for chunking/embedding/upserting docs into Qdrant |
| `tests/unit/` | Unit tests (`make test`) |
| `tests/rag_eval/` | Retrieval-quality eval fixtures + runner (`make eval`) |

## Roles (who can see what)

Defined in `app/auth/permissions.py`. `user_id` in the chat request maps to
a role via a placeholder in-memory lookup — swap it for a real identity
lookup when one exists.

- **customer** — compliance + support-docs RAG only; can only check status
  (shipment/sales-order/return), nothing internal, nothing that writes data.
- **support_agent** — every RAG domain including internal known-issues docs;
  every read-only tool; no write/approval tools.
- **manager** — everything, including write/approval tools
  (`updateShipmentEta`, `receiveAsn`, `allocateInventory`, `adjustStock`,
  `approveReturn`, etc.) — these require explicit confirmation via
  `/chat/confirm` before executing.
