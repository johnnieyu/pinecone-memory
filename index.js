import { Pinecone } from "@pinecone-database/pinecone";
import { randomUUID } from "node:crypto";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function resolveEnvVars(value) {
  if (typeof value !== "string") return value;
  return value.replace(/\$\{([^}]+)\}/g, (_, name) => process.env[name] ?? "");
}

const CAPTURE_PATTERNS = [
  /\b(prefer|always|never|like|dislike|hate|love|want|avoid)\b/i,
  /\b(decided|decision|chose|chosen|go with|pick|selected)\b/i,
  /\b(project|repo|codebase|stack|architecture)\b/i,
  /\b(use|using|switch to|migrate|adopt)\b.*\b(for|instead|over)\b/i,
  /\b(remember|note|important|keep in mind|fyi)\b/i,
  /\b(convention|pattern|style|standard|rule)\b/i,
  /\b(api key|endpoint|url|config|credentials)\b/i,
  /\b(workflow|process|routine|habit)\b/i,
];

const MIN_CAPTURE_LENGTH = 20;
const MAX_CAPTURE_LENGTH = 2000;

function shouldCapture(text) {
  if (!text || text.length < MIN_CAPTURE_LENGTH || text.length > MAX_CAPTURE_LENGTH) {
    return false;
  }
  return CAPTURE_PATTERNS.some((re) => re.test(text));
}

function detectCategory(text) {
  const lower = text.toLowerCase();
  if (/\b(prefer|always|never|like|dislike|hate|love|want|avoid)\b/.test(lower)) return "preference";
  if (/\b(decided|decision|chose|chosen|go with|selected)\b/.test(lower)) return "decision";
  if (/\b(project|repo|codebase|stack|architecture)\b/.test(lower)) return "project";
  if (/\b(bug|error|fix|debug|issue|crash|exception|stack trace)\b/.test(lower)) return "technical";
  if (/\b(is|are|was|were|has|have|the .+ is)\b/.test(lower)) return "fact";
  return "general";
}

function stripMemoryTags(text) {
  return text.replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>/g, "").trim();
}

// ---------------------------------------------------------------------------
// PineconeMemoryDB
// ---------------------------------------------------------------------------

class PineconeMemoryDB {
  constructor(config) {
    this.apiKey = resolveEnvVars(config.pineconeApiKey);
    this.indexName = config.indexName ?? "openclaw-memory";
    this.namespace = config.namespace ?? "default";
    this.deduplicationThreshold = config.deduplicationThreshold ?? 0.95;

    this.client = new Pinecone({ apiKey: this.apiKey });
    this._index = null;
    this._ready = null;
  }

  async ensureIndex() {
    if (this._ready) return this._ready;

    this._ready = (async () => {
      const { indexes } = await this.client.listIndexes();
      const exists = indexes?.some((idx) => idx.name === this.indexName);

      if (!exists) {
        throw new Error(
          `Index "${this.indexName}" not found. Create it in the Pinecone dashboard with integrated inference and a field map of text → content.`
        );
      }

      this._index = this.client.index(this.indexName).namespace(this.namespace);
    })();

    return this._ready;
  }

  async store(id, text, metadata = {}) {
    await this.ensureIndex();
    await this._index.upsertRecords([
      {
        _id: id,
        content: text,
        ...metadata,
      },
    ]);
  }

  async search(query, topK = 5, threshold = 0.3) {
    await this.ensureIndex();
    const results = await this._index.searchRecords({
      query: { topK, inputs: { text: query } },
    });

    const hits = results.result?.hits ?? [];
    return hits.filter((hit) => hit._score >= threshold);
  }

  async delete(id) {
    await this.ensureIndex();
    await this._index.deleteOne(id);
  }

  async isDuplicate(text) {
    const hits = await this.search(text, 1, this.deduplicationThreshold);
    return hits.length > 0 ? hits[0] : null;
  }
}

// ---------------------------------------------------------------------------
// Named exports for testability
// ---------------------------------------------------------------------------
export { resolveEnvVars, shouldCapture, detectCategory, stripMemoryTags, PineconeMemoryDB };

// ---------------------------------------------------------------------------
// Plugin export
// ---------------------------------------------------------------------------

export default function activate(context) {
  console.log("[pinecone-memory] activate() called");
  console.log("[pinecone-memory] context keys:", Object.keys(context));
  console.log("[pinecone-memory] context.config:", JSON.stringify(context.config, null, 2));
  const config = context.config ?? {};
  const db = new PineconeMemoryDB(config);

  const autoCapture = config.autoCapture !== false;
  const autoRecall = config.autoRecall !== false;
  const topK = config.topK ?? 5;
  const similarityThreshold = config.similarityThreshold ?? 0.3;

  // -------------------------------------------------------------------------
  // Hook: before_agent_start — Recall
  // -------------------------------------------------------------------------
  if (autoRecall) {
    context.onHook("before_agent_start", async (event) => {
      try {
        const prompt = event.prompt?.trim();
        if (!prompt || prompt.length < 5) return {};

        const hits = await db.search(prompt, topK, similarityThreshold);
        if (hits.length === 0) return {};

        const lines = hits.map((hit) => {
          const cat = hit.category ? ` [${hit.category}]` : "";
          const score = hit._score.toFixed(2);
          return `- (${score}${cat}) ${hit.content}`;
        });

        const block = [
          "<relevant-memories>",
          ...lines,
          "</relevant-memories>",
        ].join("\n");

        console.log(`[pinecone-memory] Recalled ${hits.length} memor${hits.length === 1 ? "y" : "ies"}.`);
        return { prependContext: block };
      } catch (err) {
        console.error("[pinecone-memory] Recall error:", err.message);
        return {};
      }
    });
  }

  // -------------------------------------------------------------------------
  // Hook: agent_end — Capture
  // -------------------------------------------------------------------------
  if (autoCapture) {
    context.onHook("agent_end", async (event) => {
      try {
        const messages = (event.messages ?? []).slice(-10);
        const captured = [];

        for (const msg of messages) {
          if (msg.role !== "user" && msg.role !== "assistant") continue;

          let text = typeof msg.content === "string"
            ? msg.content
            : Array.isArray(msg.content)
              ? msg.content.filter((b) => b.type === "text").map((b) => b.text).join("\n")
              : "";

          text = stripMemoryTags(text);
          if (!shouldCapture(text)) continue;

          const dup = await db.isDuplicate(text);
          if (dup) continue;

          const category = detectCategory(text);
          const id = randomUUID();

          await db.store(id, text, {
            category,
            role: msg.role,
            capturedAt: new Date().toISOString(),
          });

          captured.push({ id, category });
        }

        if (captured.length > 0) {
          console.log(
            `[pinecone-memory] Captured ${captured.length} memor${captured.length === 1 ? "y" : "ies"}: ${captured.map((c) => c.category).join(", ")}`
          );
        }
      } catch (err) {
        console.error("[pinecone-memory] Capture error:", err.message);
      }
    });
  }

  // -------------------------------------------------------------------------
  // Tool: memory_store
  // -------------------------------------------------------------------------
  context.registerTool({
    name: "memory_store",
    description:
      "Store a fact, preference, or decision in long-term memory. Use this when the user explicitly asks you to remember something.",
    parameters: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "The memory content to store.",
        },
        category: {
          type: "string",
          enum: ["preference", "decision", "project", "technical", "fact", "general"],
          description: "Category for the memory. Auto-detected if omitted.",
        },
      },
      required: ["text"],
    },
    async execute({ text, category }) {
      try {
        await db.ensureIndex();

        const dup = await db.isDuplicate(text);
        if (dup) {
          return {
            result: `Duplicate detected — a very similar memory already exists (score: ${dup._score.toFixed(2)}, id: ${dup._id}). Not stored.`,
          };
        }

        const id = randomUUID();
        const cat = category ?? detectCategory(text);

        await db.store(id, text, {
          category: cat,
          role: "tool",
          capturedAt: new Date().toISOString(),
        });

        return { result: `Memory stored (id: ${id}, category: ${cat}).` };
      } catch (err) {
        return { error: `Failed to store memory: ${err.message}` };
      }
    },
  });

  // -------------------------------------------------------------------------
  // Tool: memory_search
  // -------------------------------------------------------------------------
  context.registerTool({
    name: "memory_search",
    description:
      "Search long-term memory for relevant facts, preferences, or decisions.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query.",
        },
        limit: {
          type: "number",
          description: "Maximum results to return (default: 5).",
        },
      },
      required: ["query"],
    },
    async execute({ query, limit }) {
      try {
        const k = limit ?? topK;
        const hits = await db.search(query, k, similarityThreshold);

        if (hits.length === 0) {
          return { result: "No matching memories found." };
        }

        const formatted = hits.map((hit) => ({
          id: hit._id,
          score: hit._score.toFixed(2),
          category: hit.category ?? "unknown",
          content: hit.content,
          capturedAt: hit.capturedAt ?? null,
        }));

        return { result: JSON.stringify(formatted, null, 2) };
      } catch (err) {
        return { error: `Search failed: ${err.message}` };
      }
    },
  });

  // -------------------------------------------------------------------------
  // Tool: memory_forget
  // -------------------------------------------------------------------------
  context.registerTool({
    name: "memory_forget",
    description:
      "Delete a memory by ID, or search for and delete a matching memory.",
    parameters: {
      type: "object",
      properties: {
        memoryId: {
          type: "string",
          description: "Exact memory ID to delete.",
        },
        query: {
          type: "string",
          description: "Search query to find the memory to delete.",
        },
      },
    },
    async execute({ memoryId, query }) {
      try {
        await db.ensureIndex();

        if (memoryId) {
          await db.delete(memoryId);
          return { result: `Memory ${memoryId} deleted.` };
        }

        if (!query) {
          return { error: "Provide either memoryId or query." };
        }

        const hits = await db.search(query, 5, similarityThreshold);
        if (hits.length === 0) {
          return { result: "No matching memories found to delete." };
        }

        // Auto-delete if single high-confidence match
        if (hits.length === 1 || hits[0]._score >= 0.9) {
          const target = hits[0];
          await db.delete(target._id);
          return {
            result: `Deleted memory (id: ${target._id}, score: ${target._score.toFixed(2)}): "${(target.content ?? "").slice(0, 100)}"`,
          };
        }

        // Multiple candidates — list them
        const candidates = hits.map((hit) => ({
          id: hit._id,
          score: hit._score.toFixed(2),
          content: (hit.content ?? "").slice(0, 100),
        }));

        return {
          result: `Multiple matches found. Specify a memoryId to delete:\n${JSON.stringify(candidates, null, 2)}`,
        };
      } catch (err) {
        return { error: `Forget failed: ${err.message}` };
      }
    },
  });

  // -------------------------------------------------------------------------
  // CLI: pinecone-memory search <query> [--limit N]
  // -------------------------------------------------------------------------
  context.registerCommand({
    name: "pinecone-memory search",
    description: "Search memories from the command line.",
    parameters: [
      { name: "query", type: "string", required: true, description: "Search query" },
      { name: "--limit", type: "number", required: false, description: "Max results" },
    ],
    async execute(args) {
      try {
        const limit = args["--limit"] ?? topK;
        const hits = await db.search(args.query, limit, similarityThreshold);

        if (hits.length === 0) {
          console.log("No matching memories found.");
          return;
        }

        for (const hit of hits) {
          const cat = hit.category ? ` [${hit.category}]` : "";
          console.log(`  ${hit._score.toFixed(2)}${cat}  ${hit._id}`);
          console.log(`    ${hit.content}`);
          console.log();
        }
      } catch (err) {
        console.error("Search error:", err.message);
      }
    },
  });

  // -------------------------------------------------------------------------
  // CLI: pinecone-memory stats
  // -------------------------------------------------------------------------
  context.registerCommand({
    name: "pinecone-memory stats",
    description: "Show memory plugin status and configuration.",
    async execute() {
      try {
        await db.ensureIndex();

        console.log("[pinecone-memory] Configuration:");
        console.log(`  Index:         ${db.indexName}`);
        console.log(`  Namespace:     ${db.namespace}`);
        console.log(`  Model:         ${db.embeddingModel}`);
        console.log(`  Auto-capture:  ${autoCapture}`);
        console.log(`  Auto-recall:   ${autoRecall}`);
        console.log(`  Top-K:         ${topK}`);
        console.log(`  Threshold:     ${similarityThreshold}`);
        console.log(`  Dedup:         ${db.deduplicationThreshold}`);
      } catch (err) {
        console.error("Stats error:", err.message);
      }
    },
  });
}
