# Logistics AI System — Architecture Blueprint

This document defines how to build and extend our logistics AI system.
Any developer or AI coding agent adding a new model, MCP server, or RAG
source MUST follow this pattern. The goal: every new piece of
functionality plugs in without modifying existing code.

## Core Principle

Every layer talks to the others through an **interface (contract)**, never
a concrete implementation. Concrete implementations (Ollama, a specific
MCP server, a specific PDF country-set) are swappable plug-ins behind that
interface. If you ever find yourself editing existing orchestrator code
to add a new model/tool/doc-source, you've broken the pattern — it should
only require adding a new plug-in file + one registry entry.

## System Layers

```
┌─────────────────────────────────────────────────────────┐
│  FastAPI Layer (HTTP endpoints, request/response models) │
└───────────────────────┬───────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────┐
│              Orchestrator (the request loop)               │
│  - receives user question                                  │
│  - calls RAG registry for relevant doc chunks               │
│  - calls Model Provider with tools + context                │
│  - executes any MCP tool calls the model requests           │
│  - loops until model returns final answer                   │
│  - does NOT know which model/tool/doc-source is plugged in  │
└──────┬─────────────────┬─────────────────┬─────────────────┘
       │                 │                 │
┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
│ Model        │   │ MCP         │   │ RAG          │
│ Provider     │   │ Registry    │   │ Registry     │
│ Registry     │   │ (tools)     │   │ (knowledge)  │
└──────┬───────┘   └──────┬──────┘   └──────┬───────┘
       │                  │                 │
  Ollama / Claude    shipments-mcp     UAE compliance
  / future model     warehouse-mcp     KSA compliance
  (each a plug-in)   crm-mcp (each     Qatar compliance
                     a plug-in)        (each a plug-in)
```

## Folder Structure

```
logistics-ai-system/
├── app/
│   ├── main.py                    # FastAPI app entrypoint
│   ├── api/
│   │   └── routes/
│   │       └── chat.py            # POST /chat endpoint (supports streaming)
│   ├── orchestrator/
│   │   ├── engine.py              # the request loop (layer described above)
│   │   └── grounding.py           # output validation / grounding check
│   ├── model_providers/
│   │   ├── base.py                # ModelProvider interface (abstract class)
│   │   ├── ollama_provider.py      # implements base.py for Ollama
│   │   ├── claude_provider.py      # implements base.py for Claude API
│   │   └── registry.py            # maps a name string -> provider instance
│   ├── mcp_clients/
│   │   ├── base.py                # McpClient interface (with safety tagging)
│   │   ├── shipments_client.py    # connects to shipments MCP server
│   │   ├── warehouse_client.py     # (future) connects to warehouse MCP server
│   │   └── registry.py            # maps tool name -> which MCP client owns it
│   ├── rag/
│   │   ├── base.py                # RagSource interface
│   │   ├── vector_store.py        # Qdrant/Chroma wrapper (implementation detail)
│   │   ├── domain_sources.py      # config: domain (+ tags like country) -> collection mapping
│   │   ├── router.py              # decides which domain(s) to search per question
│   │   └── registry.py            # maps domain name -> RagSource instance
│   ├── glossary.py                # small structured terminology (order prefixes etc.)
│   ├── auth/
│   │   └── permissions.py         # role -> allowed tools/domains mapping
│   ├── logging/
│   │   └── request_logger.py      # logs model calls, tool calls, RAG queries per request_id
│   ├── config.py                  # env vars, which providers are active
│   └── schemas.py                 # Pydantic request/response models
├── mcp_servers/
│   ├── shipments/                 # our existing server.js (or python equivalent)
│   ├── warehouse/                 # future MCP server
│   └── README.md                  # how to add a new MCP server
├── ingestion/
│   ├── cli.py                     # ingest add/sync/rebuild/status/remove
│   └── manifest.json              # file hash + version tracking (audit trail)
└── tests/
    ├── rag_eval/                  # known question -> expected answer per domain
    │   ├── compliance.json
    │   ├── support-docs.json
    │   └── known-issues.json
    └── unit/                      # isolated tests per interface (model/mcp/rag/embedding/search)
```

## The Interfaces (contracts)

### 1. ModelProvider — `app/model_providers/base.py`

Every model (Ollama, Claude, future models) must implement:

```python
from abc import ABC, abstractmethod

class ModelProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict], tools: list[dict]) -> dict:
        """
        messages: full conversation history (system/user/assistant turns)
        tools: available tool schemas (from MCP registry)
        returns: { "type": "tool_call", "tool": str, "args": dict }
              or { "type": "final_answer", "text": str }
        """
        ...
```

**Rule:** the orchestrator only ever calls `.generate()`. It never knows or
cares if that's Ollama, Claude, or something added next year. Adding a new
model = one new file implementing this interface + one line in
`registry.py`. Nothing else changes.

### 2. McpClient — `app/mcp_clients/base.py`

Every MCP server connection must implement:

```python
class McpClient(ABC):
    @abstractmethod
    def list_tools(self) -> list[dict]:
        """Returns tool schemas this server exposes, each tagged with
        a `safety` level: 'read_only' or 'needs_approval'."""
        ...

    @abstractmethod
    async def call_tool(self, tool_name: str, args: dict) -> dict:
        """Executes the tool, returns the raw result."""
        ...
```

**Rule:** the orchestrator asks the registry "who owns tool X" and calls
`call_tool` on whichever client owns it. Adding a new MCP server (e.g.
warehouse-mcp) = one new client file + register its tools. No orchestrator
changes.

**Write-action safety:** any tool that changes data (e.g.
`updateShipmentEta`) must be tagged `needs_approval` in its schema. The
orchestrator never auto-executes a `needs_approval` tool call — it returns
a "confirm this action?" prompt to the user/frontend first, and only calls
`call_tool` after explicit confirmation. `read_only` tools (lookups,
searches) execute immediately, no confirmation needed. This tag lives on
the tool definition itself, not scattered as if/else logic in the
orchestrator.

### 3. RagSource — `app/rag/base.py`

RAG is NOT limited to country compliance docs. It's a general pattern for
any **knowledge domain** — compliance, support documentation, known/recurring
issues, or anything else that's document-shaped and grows over time. Each
domain is independent and pluggable.

```python
class RagSource(ABC):
    @abstractmethod
    async def search(self, query: str, filters: dict, top_k: int = 5) -> list[dict]:
        """Returns top_k relevant chunks with source metadata."""
        ...
```

**Current/planned knowledge domains (each its own collection, own ingestion
path, own lifecycle — adding or changing one never touches another):**

| Domain | Source files | Example query |
|---|---|---|
| `compliance` | PDFs, per country | "customs docs for UAE→KSA" |
| `support-docs` | Markdown/text runbooks | "how to reset a stuck shift" |
| `known-issues` | Markdown/text, recurring reported issues | "duplicate void transaction bug" |

Adding a new domain (e.g. `warehouse-sops` later) = new folder in
`ingestion/data/`, one entry in `rag/domain_sources.py`, run `ingest sync`.
No orchestrator or existing-domain changes.

**Rule:** the orchestrator queries domains by name/tag depending on
context (e.g. a support question searches `support-docs` + `known-issues`,
a customs question searches `compliance` filtered by country). Which
domain(s) to search per question is a routing decision — see "Domain
Routing" below — not something hardcoded per query.

### Domain Routing

Not every question should search every domain — that wastes latency and
can pull irrelevant chunks into context. The orchestrator (or model itself,
via a lightweight classification step) decides which domain(s) are
relevant before calling `search()`:

- Keywords/intent suggest customs, import, export, country names →
  `compliance`
- Keywords/intent suggest "how do I", "bug", "error", "not working" →
  `support-docs` + `known-issues`
- Ambiguous → search multiple domains, merge and re-rank results

This routing logic lives in one place: `app/rag/router.py` — keep it
simple (keyword/intent rules first; only reach for a model-based classifier
if simple rules prove insufficient).

### 4. EmbeddingProvider — `app/rag/embeddings/base.py`

Turns text into vectors. Also swappable, same pattern as ModelProvider.

```python
class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns one vector per input text. Must batch, not loop one-by-one."""
        ...
```

**Default implementation:** `OllamaEmbeddingProvider` using
`nomic-embed-text` — local, free, stable, good enough for compliance
retrieval. Configurable via `app/config.py` to swap to `bge-large` or a
cloud embedding API later. Both ingestion and query-time search must use
the SAME embedding provider/model — mixing embedding models between
ingest and query silently breaks retrieval quality, so this is enforced
via a single config value, not set separately in two places.

### 5. SearchStrategy — `app/rag/search/base.py`

How a query is matched against stored chunks.

```python
class SearchStrategy(ABC):
    @abstractmethod
    async def search(self, query: str, filters: dict, top_k: int) -> list[dict]:
        ...
```

Three implementations, all swappable via config:
- `DenseSearch` — pure vector/semantic similarity
- `KeywordSearch` — BM25/exact-term matching (handles article numbers, HS
  codes, regulation IDs that vector search alone tends to miss)
- `HybridSearch` **(default)** — runs both, merges/re-ranks results

**Default = Hybrid.** Compliance documents mix natural-language meaning
("what documents are needed") with exact codes/citations ("Article 12.3",
"HS Code 8471") — hybrid is the standard, stable choice for this kind of
document set. Pure dense or pure keyword remain available via config for
cases where hybrid overhead isn't needed.

## The Orchestrator Loop (pseudocode)

This is the ONE place the full request loop lives. It must stay
provider-agnostic — no `if model == "ollama"` branching anywhere in here.

```python
async def handle_question(user_id, question, tags=None, request_id=None):
    role = auth.get_role(user_id)

    # 1. Decide which knowledge domain(s) apply (compliance, support-docs, known-issues...)
    domains = rag_router.route(question, tags)
    domains = auth.filter_domains(role, domains)   # role may not see every domain

    # 2. Retrieve relevant knowledge from each applicable domain (RAG)
    chunks = []
    for domain in domains:
        rag_source = rag_registry.get(domain)
        result = await rag_source.search(question, filters=tags)
        chunks += result
        logger.log_rag_query(request_id, domain, question, result)

    # 3. Grounding check — if nothing relevant was found, say so explicitly
    if rag_router.expected_domain_but_no_hits(domains, chunks):
        return "I don't have documentation on this — you may want to check with the team."

    # 4. Build message history (from session store, e.g. Redis)
    messages = session_store.get_history(user_id)
    messages.append({"role": "system", "content": build_system_prompt(chunks)})
    messages.append({"role": "user", "content": question})

    # 5. Get only the tools this role is allowed to use
    tools = auth.filter_tools(role, mcp_registry.all_tool_schemas())

    # 6. Call the active model provider (swappable)
    provider = model_registry.get_active()
    while True:
        result = await provider.generate(messages, tools)
        logger.log_model_call(request_id, messages, result)

        if result["type"] == "final_answer":
            grounding.verify(result["text"], chunks)   # trace answer back to sources
            session_store.append(user_id, "assistant", result["text"])
            return result["text"]

        elif result["type"] == "tool_call":
            tool_schema = mcp_registry.get_schema(result["tool"])
            if tool_schema["safety"] == "needs_approval":
                return build_confirmation_prompt(result["tool"], result["args"])
                # execution resumes only after the user/frontend confirms

            client = mcp_registry.get_owner(result["tool"])
            tool_result = await client.call_tool(result["tool"], result["args"])
            logger.log_tool_call(request_id, result["tool"], result["args"], tool_result)
            messages.append({"role": "tool", "content": tool_result})
            # loop again — model sees tool result, may call another tool
            # or return final answer
```

### 6. Glossary / Terminology Lookup — `app/glossary.py` (NOT part of RAG)

Order prefixes, ID formats, internal terminology (e.g. "TRK = tracking,
ORD = order, CUST = customer") are small, stable, structured facts — not
searchable documents. Putting these into vector search is overkill and
less reliable than a direct lookup. Instead:

```python
# app/glossary.py — simple structured config, not a vector store
ID_FORMATS = {
    "TRK": {"meaning": "Tracking ID", "example": "TRK1001", "regex": r"^TRK\d{4}$"},
    "ORD": {"meaning": "Order ID", "example": "ORD1001", "regex": r"^ORD\d{4}$"},
    "CUST": {"meaning": "Customer ID", "example": "CUST1001", "regex": r"^CUST\d{4}$"},
}
```

This gets injected directly into the system prompt (small, cheap, always
available) rather than retrieved via search — same reasoning as the ID-format
example from earlier in this project. If this list grows very large
(dozens+ of terms), it can graduate into its own lightweight RAG domain,
but for a handful of stable prefixes, direct injection is simpler and more
reliable than search.

**Rule:** if a piece of company knowledge is small and rarely changes →
glossary/config. If it's document-shaped and grows over time → RAG domain.

Location: `ingestion/cli.py`. Run with `python -m ingestion.cli <command>`.
Works identically across every domain (compliance, support-docs,
known-issues, future ones) — domain is just a parameter.

```
ingest add       --domain compliance --tag country=UAE --path ./data/compliance/uae/
ingest add       --domain support-docs --path ./data/support-docs/
ingest add       --domain known-issues --path ./data/known-issues/
ingest rebuild   --domain compliance --tag country=UAE    # force full re-embed, one slice
ingest rebuild   --all                                     # force full re-embed, everything
ingest sync                                                 # scan ALL domains, only process changed/new files (default, safe to run often)
ingest status                                                # show manifest: per-domain file counts, last embedded time, pending changes
ingest remove    --domain known-issues --file old_bug.md     # remove a specific doc + its vectors
```

**Change detection (the manifest):** every ingested file's path + content
hash + embedding-model-used is recorded in a manifest (`ingestion/manifest.json`
or a small SQLite table). On `sync`:
1. Walk the configured PDF folders
2. Compare each file's current hash against the manifest
3. New file → chunk + embed + store, add to manifest
4. Changed file (hash differs) → delete its old vectors, re-chunk + re-embed, update manifest
5. Unchanged file → skip entirely (this is what avoids re-processing all 100 PDFs every time)
6. File removed from folder → flag in `status`, don't auto-delete vectors without `remove` confirmation

**Rule:** `sync` is the command you run routinely (e.g. after adding new
docs, or on a schedule/cron). `rebuild` is the manual escape hatch when you
change the embedding model itself or suspect the vector store is out of
sync — it forces a full re-embed since old vectors from a different
embedding model are not compatible with new ones.

## Rules for Adding New Functionality (plug-and-play checklist)

**Adding a new model (e.g. Qwen, GPT):**
1. Create `app/model_providers/qwen_provider.py` implementing `ModelProvider`
2. Register it in `registry.py`
3. Switch active model via config/env var — no other file touched

**Adding a new MCP server (e.g. warehouse-mcp):**
1. Build the MCP server itself (standalone, like our shipments example)
2. Create `app/mcp_clients/warehouse_client.py` implementing `McpClient`
3. Register it in `mcp_clients/registry.py`
4. Tools become instantly available to the orchestrator — no orchestrator changes

**Adding a new country's compliance docs (existing domain, new slice):**
1. Drop PDFs in `ingestion/data/compliance/<country>/`
2. Run `ingest sync` (or `ingest add --domain compliance --tag country=<country> --path ...`)
3. No code changes — `domain_sources.py` config + tags pick it up automatically

**Adding a whole new knowledge domain (e.g. support-docs, known-issues):**
1. Create `ingestion/data/<domain-name>/` and drop markdown/text files in it
2. Register the domain in `rag/domain_sources.py` (name, file types, chunking rules)
3. Add a routing rule in `rag/router.py` (which questions should search this domain)
4. Run `ingest sync` — no orchestrator or other-domain changes needed

**Adding a new API endpoint:**
1. Add route in `app/api/routes/`
2. It calls the orchestrator — never talks to model/MCP/RAG directly

## Non-Negotiable Constraints

- Orchestrator code must never import a specific model SDK, MCP client, or
  vector DB library directly — only the interfaces/registries.
- Every plug-in (model, MCP client, RAG source) is independently testable
  in isolation without the orchestrator running.
- Session/conversation history lives outside the orchestrator (Redis or
  similar) — orchestrator is stateless per call.
- Config (which model is active, which MCP servers are registered, which
  countries have RAG sources) lives in one place: `app/config.py` /
  environment variables — never hardcoded in logic files.
- All cross-cutting concerns (logging tool calls, auth, rate limiting)
  live in FastAPI middleware — not scattered inside business logic.

## Default Configuration (stable out-of-the-box, all swappable)

| Component | Default | Configurable alternatives |
|---|---|---|
| Chat model | Ollama (llama3.1) | Claude API, future models |
| Embedding model | Ollama (nomic-embed-text) | bge-large, cloud embedding API |
| Search strategy | Hybrid (dense + keyword) | Dense-only, keyword-only |
| Vector DB | Qdrant | Chroma, pgvector, Milvus |
| Session store | Redis | — |

All of these are set in `app/config.py` / environment variables, never
hardcoded in business logic. Everything works out of the box with Ollama
end-to-end (chat + embeddings) with zero cloud dependency, and can be
swapped piece by piece as needs grow.

## Cross-Cutting Concerns

These live in FastAPI middleware / dedicated modules — never scattered
inside orchestrator or plug-in business logic.

### Output Validation (Grounding Check)

Before returning a RAG-backed answer, verify the model's answer actually
relates to the retrieved chunks — don't just trust it blindly:
- If `search()` returned zero relevant chunks (below a similarity
  threshold), the orchestrator instructs the model to say "I don't have
  documentation on this" rather than answer from general knowledge.
- Log the retrieved chunk IDs alongside the final answer, so any answer
  can be traced back to its source document for audit/debugging.
- This lives in `app/orchestrator/grounding.py` — a lightweight check, not
  a second full model call, unless quality issues later prove it's needed.

### Auth & Permissions

- Every API request carries a user identity + role (e.g. `support_agent`,
  `customer`, `manager`).
- `app/auth/` resolves role → which MCP tools and which RAG domains that
  role may access (e.g. a customer-facing bot should never see
  `known-issues` internal bug docs).
- Enforced once, centrally, in FastAPI middleware — individual tools/domains
  don't each re-implement permission checks.

### Logging & Observability

- `app/logging/` — every model call, tool call (with args + result), and
  RAG query (with which chunks were retrieved) is logged with a shared
  `request_id` so a full conversation can be reconstructed and debugged.
- This is what makes "why did it answer this way" answerable later —
  treat it as required, not optional, from day one.

### Document Versioning & Audit Trail

- Compliance docs need an audit trail: when a PDF is replaced, keep the
  old version's vectors tagged with an `effective_until` date rather than
  deleting immediately — so past answers remain explainable ("this was
  correct per the rules at the time").
- The ingestion manifest (from the CLI section) already tracks file
  hash + timestamp per version — this is a natural extension of it, not a
  new system.

### RAG Evaluation Set

- Maintain a small `tests/rag_eval/<domain>.json` file per domain: known
  question → expected answer/source-doc pairs.
- Run this eval whenever you change embedding model, chunking strategy, or
  search strategy (dense/hybrid/keyword) — confirms retrieval quality
  didn't regress before rolling the change out.

### Testing the Plug-in Contracts

- Each interface (`ModelProvider`, `McpClient`, `RagSource`,
  `EmbeddingProvider`, `SearchStrategy`) gets its own unit tests in
  `tests/`, testable in isolation with mock inputs — no orchestrator or
  live services needed to verify a new plug-in works correctly.

### Streaming Responses

- FastAPI endpoint should support streaming (Server-Sent Events or chunked
  response) so the user sees the answer appear as it's generated, rather
  than waiting for the full response — most model providers (Ollama,
  Claude API) support streaming natively; the `ModelProvider` interface's
  `generate()` should have a `stream: bool` option from the start, even if
  not used immediately, to avoid a breaking interface change later.

### Multi-language Support

- If support tickets/compliance docs exist in Arabic as well as English:
  the embedding model must support multilingual text (verify
  `nomic-embed-text` or switch to a multilingual embedding model if
  needed), and chunking/language metadata should be tagged per document
  so results can be filtered or translated consistently.
- Decide early whether responses should match the query's language or
  always respond in a fixed language — this affects the system prompt.

## Performance Notes (from earlier decisions)

- Model serving: Ollama for dev, vLLM for production concurrency
- Vector DB: Qdrant
- Caching: Redis — cache repeated compliance questions before hitting RAG+model
- Orchestration: FastAPI (async throughout — no blocking calls in the loop)
- Embedding + search: batch-embed on ingestion (never one-by-one), hybrid
  search re-ranking adds a small latency cost vs. pure dense — acceptable
  tradeoff for the accuracy gain on compliance-style documents
