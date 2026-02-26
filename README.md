# openclaw-memory-pinecone

Long-term memory for [OpenClaw](https://github.com/openclaw/openclaw) agents, powered by [Pinecone](https://www.pinecone.io/) with integrated inference.

Your agent forgets everything between sessions. This plugin fixes that. It watches conversations, extracts what matters, and brings it back when relevant — automatically. No separate embedding service required — Pinecone handles embeddings server-side.

## How it works

**Auto-Recall** — Before the agent responds, the plugin searches Pinecone for memories that match the current prompt and injects them into context via `<relevant-memories>` tags.

**Auto-Capture** — After the agent responds, the plugin scans the last 10 messages for facts, preferences, and decisions. Matching text is deduplicated, categorized, and stored automatically.

Both run silently. No prompting, no manual calls.

## Setup

### Install via ClawHub (recommended)

```bash
openclaw plugins install openclaw-memory-pinecone
```

### Manual install

```bash
cd ~/.openclaw/plugins
git clone https://github.com/your-user/openclaw-memory-pinecone.git memory-pinecone
cd memory-pinecone
npm install
```

### Requirements

- Node.js >= 18
- A [Pinecone](https://www.pinecone.io/) API key (free tier works)

### Configure

Get an API key from [app.pinecone.io](https://app.pinecone.io), then add to your `openclaw.json`:

```json5
{
  "plugins": {
    "installs": {
      "openclaw-memory-pinecone": {
        "source": "./plugins/memory-pinecone",  // or the path where you cloned the repo
        "enabled": true,
        "config": {
          "pineconeApiKey": "${PINECONE_API_KEY}"
        }
      }
    }
  }
}
```

If you installed via ClawHub, omit the `source` field — it's set automatically.

You must create the Pinecone index before using the plugin. The plugin will throw a clear error if the index doesn't exist.

### Create your Pinecone index

1. Go to [app.pinecone.io](https://app.pinecone.io) and create a new **serverless** index.
2. Under **Embedding**, choose **Integrated (Inference)** and select a model (e.g. `multilingual-e5-large`).
3. Set the field map so the `text` source field maps to `content`.
4. Note the index name and set it in your config (defaults to `"openclaw-memory"`):

```json5
"config": {
  "pineconeApiKey": "${PINECONE_API_KEY}",
  "indexName": "my-custom-index"
}
```

#### Using namespaces

All memories are stored in a single namespace (`"default"`). To isolate memories per user, project, or machine, set the `namespace` option:

```json5
"config": {
  "pineconeApiKey": "${PINECONE_API_KEY}",
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
| `topK` | `number` | `5` | Max memories per recall |
| `similarityThreshold` | `number` | `0.3` | Min similarity score (0–1) for search results |
| `deduplicationThreshold` | `number` | `0.95` | Min similarity to consider a memory a duplicate |

## How capture works

Not every message is stored. The plugin uses pattern matching to detect text worth keeping:

- **Preferences** — "I prefer", "always use", "never", "avoid"
- **Decisions** — "decided to", "go with", "chose"
- **Project context** — "the project uses", "our stack", "architecture"
- **Technical notes** — "remember", "important", "keep in mind"
- **Conventions** — "convention", "pattern", "style guide"
- **Config references** — "API key", "endpoint", "credentials"

Messages shorter than 20 characters or longer than 2,000 characters are ignored. Duplicates (similarity >= 0.95) are automatically skipped.

Each stored memory is auto-categorized as one of: `preference`, `decision`, `project`, `technical`, `fact`, or `general`.

## Development

```bash
npm install
npm test        # run tests (vitest)
```

## License

MIT
