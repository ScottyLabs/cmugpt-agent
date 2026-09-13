# Bark Agent

Bark Agent is the backend service for Bark, the campus assistant for Carnegie
Mellon University built by ScottyLabs. It receives chat messages from the Bark
web application, answers them with a language model and a set of campus data
tools, applies safety and accuracy checks to each answer, and maintains
long-term memory for each user.

## Overview

Bark consists of two services.

- The Surface ([cmugpt-surface](https://git.cmu.dev/ScottyLabs/cmugpt-surface))
  is the web application and its server. It authenticates users, stores chats,
  and forwards each message to this service.
- The Agent (this repository) processes each message. It selects the campus
  tools the question requires, runs a LangGraph agent against models served
  through OpenRouter, validates the result, and streams the answer back to the
  Surface.

Campus data comes from the CMU MCP server, which publishes tools for maps,
courses, dining, and the student guide over the Model Context Protocol.
Per-user memory is stored in PostgreSQL with the pgvector extension.

```
┌───────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
│  Browser  │ ──► │  Surface                 │ ──► │  Bark Agent              │
│           │     │  web app and API server  │     │  this repository         │
└───────────┘     └──────────────────────────┘     └─────────────┬────────────┘
                                                                 │
           ┌──────────────────────────┬──────────────────────────┘
           ▼                          ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  OpenRouter          │   │  CMU MCP server      │   │  PostgreSQL          │
│  language models     │   │  campus data tools   │   │  memory (pgvector)   │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

## Request lifecycle

1. Validation. The Surface posts the message, the prior turns of the chat, and
   a hashed user identifier. The service enforces request size limits, checks
   the user's daily token budget, and screens the message with OpenAI's
   moderation endpoint before any model call.
2. Planning. `planning.py` determines what the turn requires: which tool
   groups to bind, whether the `remember` and `forget` tools are needed, and
   whether memory recall should run. Conversational messages bind no data
   tools. The map tool is bound on every turn unless the user has disabled
   maps, since the model decides whether a map belongs in the answer.
3. Execution. `graph.py` runs a LangGraph graph: recall relevant facts about
   the user, invoke the model, execute any tool calls, and repeat until the
   model produces a final answer. Tool output is wrapped as untrusted data so
   that it cannot inject instructions.
4. Verification. `guards.py` and the `maps/` package check the finished
   answer without a model. The model's map selection is validated against the
   building catalog, incorrect claims that a lookup failed are repaired,
   secrets and system prompt text are removed, and tool usage is disclosed
   accurately.
5. Delivery. The answer is streamed to the Surface as Server-Sent Events.
   Once the answer is complete, a background task extracts durable facts
   about the user from the exchange and stores them for future turns.

Memory holds only extracted facts and facts the user explicitly asks Bark to
remember. Raw chat turns are never stored. Users can view and delete their
facts through the Surface, and each user's memory is isolated by identifier.

## Project structure

```
cmugpt-agent/
│
├── src/cmugpt/
│   │
│   ├── api/                    # HTTP layer
│   │   ├── server.py           #   FastAPI app: lifespan, CORS, error envelope, routers
│   │   ├── deps.py             #   Bearer-token and body-size checks shared by routes
│   │   └── routes/
│   │       ├── agent.py        #   /agent/respond, /agent/respond/stream, /agent/title
│   │       ├── memory.py       #   /memory/*
│   │       └── health.py       #   /api/health
│   │
│   ├── settings.py             # All environment variables
│   ├── schema.py               # Request and response models
│   ├── planning.py             # Per-turn tool and memory selection
│   ├── graph.py                # LangGraph control flow
│   ├── prompts.py              # System prompt construction
│   ├── llm.py                  # OpenRouter model client factory
│   ├── mcp_tools.py            # MCP tool discovery and group filtering
│   ├── guards.py               # Deterministic output checks
│   ├── moderation.py           # OpenAI moderation for input and output
│   ├── token_limits.py         # Per-user daily token budget (SQLite)
│   ├── title.py                # Chat title generation
│   │
│   ├── memory/                 # Per-user long-term memory
│   │   ├── store.py            #   LangGraph store: Postgres with pgvector, or in-memory
│   │   ├── facts.py            #   Recall, save, forget
│   │   ├── tools.py            #   The remember and forget tools exposed to the model
│   │   ├── extraction.py       #   Background fact extraction
│   │   └── manage.py           #   List, delete, clear
│   │
│   └── maps/                   # Campus map support
│       ├── buildings.py        #   Building catalog and aliases
│       ├── buildings.json      #   Catalog data
│       ├── tool.py             #   The maps_show_map tool
│       └── inference.py        #   Map validation and URL construction
│
├── tests/unit/                 # Offline tests, run by CI
├── evals/                      # Live evaluations, run manually
│
├── pyproject.toml              # Dependencies, entry point, tool configuration
├── Procfile                    # Start command from the earlier Railway deployment, unused by Kennel
├── devenv.nix                  # Local environment and Kennel settings
├── flake.nix                   # Nix build
├── secretspec.toml             # Secret declarations
└── .env.example                # Environment variable template
```

## Requirements

- Python 3.12. uv installs it if it is not present.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- PostgreSQL with the pgvector extension, for persistent memory. Optional for
  local development.
- [devenv](https://devenv.sh) with direnv, for the same environment CI and
  production use. Optional. It provides PostgreSQL with pgvector, exports
  `DATABASE_URL`, and installs the project's git hooks (ruff, ty, formatting,
  TOML checks, and secret scanning).

## Installation

```sh
git clone https://git.cmu.dev/ScottyLabs/cmugpt-agent.git
cd cmugpt-agent
uv sync
cp .env.example .env
```

Edit `.env` and set the variables described below.

## Configuration

All configuration is read from environment variables by `settings.py`.
`.env.example` documents each variable and its default.

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Yes | Chat, memory extraction, and chat titles |
| `MCP_SERVER_URL` | Yes | Base URL of the CMU MCP server |
| `OPENAI_API_KEY` | Recommended | Embeddings for memory search and the moderation endpoint. Without it, recall orders facts by recency and moderation is skipped |
| `AGENT_SHARED_SECRET` | In production | Bearer token the Surface presents on every request. Empty disables authentication |
| `DATABASE_URL` | In production | PostgreSQL connection string. Unset selects an in-memory store that is cleared on restart |
| `ALLOWED_ORIGINS` | No | Comma-separated browser origins for CORS. Default `https://cmugpt.com` |
| `PORT` | No | Listening port. Default `5000` |
| `TITLE_MODEL` | No | Model for chat titles. Default `qwen/qwen3.7-flash` |
| `MEMORY_EXTRACTION_MODEL` | No | Model for background fact extraction. Default `qwen/qwen3.7-flash` |
| `TOKEN_USAGE_DB` | No | SQLite file for the daily token budget. Default `/tmp/cmugpt_token_usage.sqlite3` |

### Memory database

For persistent memory, create a PostgreSQL database with the pgvector
extension and point `DATABASE_URL` at it. The service creates its own schema
and tables on first start.

```sh
createdb cmugpt_agent
psql -d cmugpt_agent -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

```
DATABASE_URL=postgresql:///cmugpt_agent?host=/tmp
```

With devenv, this step is unnecessary. The shell provides a local PostgreSQL
instance with pgvector and exports `DATABASE_URL`.

## Running the service

```sh
uv run cmugpt-agent
```

The service listens on port 5000 by default. Confirm it is healthy:

```sh
curl -s localhost:5000/api/health
```

```json
{"status": "ok", "memory": {"backend": "postgres", "initialized": true, "semantic_search": true, "embedding_model": "text-embedding-3-large", "ready": true}}
```

`backend` reports `in-memory` when `DATABASE_URL` is unset, and
`semantic_search` is `false` when `OPENAI_API_KEY` is unset. The endpoint
returns HTTP 503 with `"status": "degraded"` when the memory store cannot be
queried.

## API

When `AGENT_SHARED_SECRET` is set, every route except `/api/health` requires
the header `Authorization: Bearer <secret>`.

| Route | Description |
| --- | --- |
| `POST /agent/respond` | Returns the complete answer as a JSON object |
| `POST /agent/respond/stream` | Returns the answer as Server-Sent Events |
| `POST /agent/title` | Generates a short title from a chat's first message |
| `GET /memory/{user_id}` | Lists a user's stored facts, with search and paging |
| `DELETE /memory/{user_id}/items/{kind}/{item_id}` | Deletes one fact |
| `DELETE /memory/{user_id}` | Deletes all facts for a user |
| `GET /api/health` | Service status and active memory backend |

Example request:

```sh
curl -s localhost:5000/agent/respond \
  -H 'content-type: application/json' \
  -d '{"query": "What is open for lunch near Gates?", "user_id": "example"}'
```

Request fields for `/agent/respond` and `/agent/respond/stream`:

| Field | Description |
| --- | --- |
| `query` | The user's message. Required. At most 8,000 characters |
| `user_id` | Identifier for memory and the token budget. The Surface sends a hash of the authenticated user |
| `message_history` | Prior turns as `{"role", "content"}` objects. The last 40 are used |
| `model` | OpenRouter model identifier. Default `openai/gpt-5.6-luna` |
| `disabled_tools` | Tool groups the user has switched off: `maps`, `courses`, `eats`, `guide` |

The streaming endpoint emits `status` events while tools run, `delta` events
carrying text as it is generated, a `map` event when a campus map accompanies
the answer, a `memory` event when a fact is saved or removed, and a final
`done` event containing the complete response object. An `error` event
terminates a failed turn.

Each user is limited to one million tokens per day. Requests beyond that
limit receive HTTP 429.

## Testing

```sh
DATABASE_URL="" uv run pytest    # offline unit tests, as run by CI
uv run pytest evals              # live evaluations
```

Unit tests in `tests/unit/` run with the model replaced by a stub and an
in-memory store. They are deterministic and fail only when code is incorrect.

Evaluations in `evals/` send real questions to the configured model and MCP
server and check the behavior of the answers: tool usage, refusal of prompt
injection, and absence of fabricated details. They require `OPENROUTER_API_KEY`
and `MCP_SERVER_URL`, incur API costs, and skip automatically when the keys
are absent. They are not part of the default `pytest` run.

## Development

```sh
uv run ruff format    # format
uv run ruff check     # lint. The project configuration applies fixes.
uv run ty check       # type check
```

Entering the devenv shell installs the project's git hooks. CI runs the same
checks over the whole repository.


## Deployment

Production runs on [Kennel](https://git.cmu.dev/ScottyLabs/kennel), the
ScottyLabs deployment platform. Kennel builds the `agent` package defined in
`flake.nix` and runs its `cmugpt-agent` entry point as a systemd unit, with
`PORT`, `DATABASE_URL`, and the secrets from the `prod` profile of
`secretspec.toml` injected as environment variables. Pushes to `main` on
git.cmu.dev trigger a deployment, and each pull request receives a preview
deployment.
- https://api.cmugpt-agent.scottylabs.org (custom domain)
- https://cmugpt-agent-agent-main.scottylabs.net (default Kennel URL)

Verify that a change builds before pushing:

```sh
SECRETSPEC_PROVIDER=dotenv://.env devenv build scottylabs.kennel.config
nix build .#packages.x86_64-linux.agent
```

Production secrets are stored in OpenBao and managed with secretspec.
Membership in the `cmugpt-agent-admins` group and `bao login -method=oidc`
are required:

```sh
secretspec set -P prod OPENROUTER_API_KEY
secretspec set -P prod OPENAI_API_KEY
secretspec set -P prod MCP_SERVER_URL
secretspec set -P prod AGENT_SHARED_SECRET
secretspec check -P prod
```

`DATABASE_URL` is not stored as a secret. Kennel injects it from its managed
PostgreSQL instance.

The service has a production startup check that refuses to run unless
`DATABASE_URL` is set and `AGENT_SHARED_SECRET` is at least 32 characters
long. The check activates only when `AGENT_ENV`, `APP_ENV`, `ENVIRONMENT`, or
`SECRETSPEC_PROFILE` is set to `production` or `prod`. Kennel sets none of
these, so the check is inactive in the current deployment.

Memory search uses a pgvector `halfvec(3072)` index sized for OpenAI's
`text-embedding-3-large`. On startup the service verifies that an existing
`store_vectors` table matches this shape and refuses to run against a table
built for a different embedding model. To recover, drop the `store_vectors`
and `vector_migrations` tables in the `agent_memory` schema and restart. Any
memory that must be retained has to be re-indexed.

## Code style

Do not disable `ruff` or `ty` rules for the whole project. Where a line
requires an exception, use the narrowest directive that applies, in this
order of preference.

For `ty`:

1. `# ty: ignore[<rule>]` for a single rule
2. `# ty: ignore[rule1, rule2]` for several rules
3. `# type: ignore` or `# type: ignore[<rule>]` for all violations on the line, even when a rule is named
4. `@typing.no_type_check` on a function

For `ruff`:

1. `# noqa: <rule>` for a single rule
2. `# noqa: rule1, rule2` for several rules
3. `# noqa` for all violations on the line
4. `# ruff: noqa: <rule>` for a single rule across a file
5. `# ruff: noqa` for all violations across a file

## Related repositories

- [cmugpt-surface](https://git.cmu.dev/ScottyLabs/cmugpt-surface): the web application and its server.
- [kennel](https://git.cmu.dev/ScottyLabs/kennel): the deployment platform.

## License

Apache License 2.0. See [LICENSE](LICENSE).
