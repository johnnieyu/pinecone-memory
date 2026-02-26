# openclaw-memory-pinecone

Long-term memory for [OpenClaw](https://github.com/openclaw/openclaw) agents, powered by [Pinecone](https://www.pinecone.io/) with integrated inference.

Your agent forgets everything between sessions. This plugin fixes that. It watches conversations, extracts what matters, and brings it back when relevant — automatically. No separate embedding service required — Pinecone handles embeddings server-side.

## How it works

**Auto-Recall** — Before the agent responds, the plugin searches Pinecone for memories that match the current prompt and injects them into context via `<relevant-memories>` tags.

**Auto-Capture** — After the agent responds, the plugin extracts facts, preferences, and decisions from the last 10 messages and stores them. Two capture modes are available:

- **Heuristic** (default) — Fast regex-based pattern matching. No external API calls required.
- **LLM** — Uses OpenAI (gpt-4o-mini by default) to extract concise facts and intelligently reconcile them against existing memories with ADD/UPDATE/DELETE/NONE decisions. Produces significantly more accurate, concise memories. Falls back to heuristic on any LLM failure.

Both hooks run silently. No prompting, no manual calls.

## Setup

There are three steps: create the Pinecone index, install the plugin, and configure `openclaw.json`.

### Step 1: Create your Pinecone index

The plugin does **not** auto-create indexes. You must create one manually before enabling the plugin.

1. Go to [app.pinecone.io](https://app.pinecone.io) and create a new **serverless** index.
2. Choose a cloud/region (e.g. `aws` / `us-east-1`).
3. Under **Embedding**, select **Integrated (Inference)** and pick a model (e.g. `llama-text-embed-v2`).
4. Set the **source field** to `content`. This is the field the plugin writes memory text into.
5. Note the index name you chose (e.g. `openclaw-woodhouse`).

Your index record shape will look like this:

```json
{
  "_id": "doc1",
  "content": "the quick brown fox jumped over the lazy dog"
}
```

The plugin writes records in this exact shape — `_id` + `content` + metadata fields (`category`, `role`, `capturedAt`).

### Step 2: Install the plugin

#### Via ClawHub (recommended)

```bash
openclaw plugins install openclaw-memory-pinecone
```

#### Manual install

```bash
git clone https://github.com/your-user/openclaw-memory-pinecone.git /path/to/pinecone-memory
cd /path/to/pinecone-memory
npm install
```

#### Requirements

- Node.js >= 18
- A [Pinecone](https://www.pinecone.io/) API key (free tier works)

### Step 3: Configure `openclaw.json`

Your `~/.openclaw/openclaw.json` needs two sections for the plugin: an install record under `plugins.installs` and an entry under `plugins.entries`.

#### If installed via ClawHub

ClawHub writes the `plugins.installs` record automatically. You only need to configure `plugins.entries`:

```json5
{
  "plugins": {
    "entries": {
      "openclaw-memory-pinecone": {
        "enabled": true,
        "config": {
          "pineconeApiKey": "${PINECONE_API_KEY}",
          "indexName": "openclaw-woodhouse"
        }
      }
    }
  }
}
```

#### If installed manually

You need both `plugins.installs` (so OpenClaw knows where the plugin lives) and `plugins.entries` (to enable and configure it).

The `source` field must be one of: `"npm"`, `"archive"`, or `"path"`. For a local clone, use `"path"`:

```json5
{
  "plugins": {
    "installs": {
      "openclaw-memory-pinecone": {
        "source": "path",
        "sourcePath": "/path/to/pinecone-memory",
        "installPath": "/Users/you/.openclaw/extensions/openclaw-memory-pinecone",
        "version": "1.0.0",
        "resolvedName": "openclaw-memory-pinecone"
      }
    },
    "entries": {
      "openclaw-memory-pinecone": {
        "enabled": true,
        "config": {
          "pineconeApiKey": "${PINECONE_API_KEY}",
          "indexName": "openclaw-woodhouse"
        }
      }
    }
  }
}
```

#### Full config example (heuristic mode)

```json5
{
  "plugins": {
    "installs": {
      "openclaw-memory-pinecone": {
        "source": "path",
        "sourcePath": "/Users/you/repos/pinecone-memory",
        "installPath": "/Users/you/.openclaw/extensions/openclaw-memory-pinecone",
        "version": "1.0.0",
        "resolvedName": "openclaw-memory-pinecone"
      }
    },
    "entries": {
      "openclaw-memory-pinecone": {
        "enabled": true,
        "config": {
          "pineconeApiKey": "${PINECONE_API_KEY}",
          "indexName": "openclaw-woodhouse",
          "namespace": "default",
          "autoRecall": true,
          "autoCapture": true,
          "topK": 5,
          "similarityThreshold": 0.3,
          "deduplicationThreshold": 0.95
        }
      }
    }
  }
}
```

#### Full config example (LLM mode)

LLM mode uses OpenAI to extract and reconcile facts. This produces higher-quality memories but requires an OpenAI API key:

```json5
{
  "plugins": {
    "installs": {
      "openclaw-memory-pinecone": {
        "source": "path",
        "sourcePath": "/Users/you/repos/pinecone-memory",
        "installPath": "/Users/you/.openclaw/extensions/openclaw-memory-pinecone",
        "version": "1.0.0",
        "resolvedName": "openclaw-memory-pinecone"
      }
    },
    "entries": {
      "openclaw-memory-pinecone": {
        "enabled": true,
        "config": {
          "pineconeApiKey": "${PINECONE_API_KEY}",
          "indexName": "openclaw-woodhouse",
          "namespace": "default",
          "autoRecall": true,
          "autoCapture": true,
          "captureMode": "llm",
          "openaiApiKey": "${OPENAI_API_KEY}",
          "llmModel": "gpt-4o-mini",
          "topK": 5,
          "similarityThreshold": 0.3,
          "deduplicationThreshold": 0.95
        }
      }
    }
  }
}
```

#### Environment variables

Set your API keys as environment variables so the `${...}` references resolve:

```bash
export PINECONE_API_KEY="pcsk_..."
export OPENAI_API_KEY="sk-..."   # only needed for captureMode: "llm"
```

Add these to your `~/.zshrc` or `~/.bashrc` to persist across sessions.

### Using namespaces

All memories are stored in a single namespace (`"default"`). To isolate memories per user, project, or machine, set the `namespace` option:

```json5
"config": {
  "pineconeApiKey": "${PINECONE_API_KEY}",
  "indexName": "openclaw-woodhouse",
  "namespace": "work-laptop"
}
```

Multiple namespace configs can share the same index without interfering with each other.

## Agent tools

The agent gets three tools it can call during conversations:

| Tool | Description |
|------|-------------|
| `memory_store` | Save a fact, preference, or decision to long-term memory |
| `memory_search` | Search memories by natural language query |
| `memory_forget` | Delete a memory by ID, or search-and-delete by query |

## CLI

```bash
# Search memories
openclaw pinecone-memory search "what framework does the user prefer"

# Show plugin config and status
openclaw pinecone-memory stats
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pineconeApiKey` | `string` | — | **Required.** Pinecone API key (supports `${PINECONE_API_KEY}`) |
| `indexName` | `string` | `"openclaw-memory"` | Pinecone index name (must already exist) |
| `namespace` | `string` | `"default"` | Namespace within the index |
| `autoRecall` | `boolean` | `true` | Inject relevant memories before each turn |
| `autoCapture` | `boolean` | `true` | Store facts after each turn |
| `captureMode` | `string` | `"heuristic"` | `"heuristic"` for regex-based capture, `"llm"` for LLM-driven extraction |
| `openaiApiKey` | `string` | — | OpenAI API key (supports `${OPENAI_API_KEY}`). Required when `captureMode` is `"llm"` |
| `llmModel` | `string` | `"gpt-4o-mini"` | OpenAI model for fact extraction and reconciliation |
| `topK` | `number` | `5` | Max memories per recall |
| `similarityThreshold` | `number` | `0.3` | Min similarity score (0-1) for search results |
| `deduplicationThreshold` | `number` | `0.95` | Min similarity to consider a memory a duplicate |

## How capture works

### Heuristic mode (default)

Not every message is stored. The plugin uses pattern matching to detect text worth keeping:

- **Preferences** — "I prefer", "always use", "never", "avoid"
- **Decisions** — "decided to", "go with", "chose"
- **Project context** — "the project uses", "our stack", "architecture"
- **Technical notes** — "remember", "important", "keep in mind"
- **Conventions** — "convention", "pattern", "style guide"
- **Config references** — "API key", "endpoint", "credentials"

Messages shorter than 20 characters or longer than 2,000 characters are ignored. Duplicates (similarity >= 0.95) are automatically skipped. Each stored memory is auto-categorized as one of: `preference`, `decision`, `project`, `technical`, `fact`, or `general`.

### LLM mode

When `captureMode` is `"llm"`, the plugin uses a 2-step pipeline inspired by [mem0](https://github.com/mem0ai/mem0):

1. **Fact extraction** — The conversation is sent to an LLM (gpt-4o-mini by default) which extracts concise, standalone facts about developer preferences, technical decisions, project details, and workflow conventions.
2. **Memory reconciliation** — Each extracted fact is compared against existing memories via Pinecone similarity search. A second LLM call decides for each fact whether to ADD (new info), UPDATE (refine existing), DELETE (contradicts existing), or NONE (already captured).

This produces more accurate, concise memories than heuristic mode. LLM calls use `temperature: 0` and structured JSON output for deterministic results. If any LLM call fails, the plugin automatically falls back to heuristic capture — memory capture is non-critical and should never block the agent.

## Troubleshooting

### `plugins.installs.*.source: Invalid input`

The `source` field only accepts `"npm"`, `"archive"`, or `"path"`. If you installed manually, use `"path"`:

```json
"source": "path"
```

### `Index "..." not found`

The plugin does not auto-create indexes. Create one in the [Pinecone dashboard](https://app.pinecone.io) with:
- Type: **Serverless**
- Embedding: **Integrated (Inference)** with any supported model
- Source field: `content`

### Memories not being captured

Check that `autoCapture` is `true` in your config and that your messages match at least one capture pattern. Short messages (< 20 chars) and very long messages (> 2000 chars) are ignored.

### Memories not being recalled

Check that `autoRecall` is `true` and that your prompt is at least 5 characters. Lower `similarityThreshold` if results are too strict.

## Development

```bash
npm install
npm test        # run tests (vitest)
```

## License

MIT
