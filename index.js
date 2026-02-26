import { Pinecone } from "@pinecone-database/pinecone";
import { randomUUID } from "node:crypto";
import OpenAI from "openai";

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

function extractHitContent(hit) {
  if (!hit || typeof hit !== "object") return "";
  const raw =
    hit.content ??
    hit.fields?.content ??
    hit.metadata?.content ??
    hit.record?.content ??
    hit.values?.content ??
    "";
  if (typeof raw === "string") return raw.trim();
  if (raw == null) return "";
  return String(raw).trim();
}

function normalizeFact(text) {
  return text
    .replace(/^[\-•\*\d\.\)\s]+/, "")
    .replace(/\s+/g, " ")
    .trim();
}

function splitSentences(text) {
  return text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => normalizeFact(s))
    .filter(Boolean);
}

function tokenSet(text) {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 2)
  );
}

function similarity(a, b) {
  const aSet = tokenSet(a);
  const bSet = tokenSet(b);
  if (aSet.size === 0 || bSet.size === 0) return 0;
  let intersection = 0;
  for (const token of aSet) if (bSet.has(token)) intersection += 1;
  const union = new Set([...aSet, ...bSet]).size;
  return union === 0 ? 0 : intersection / union;
}

function isContradiction(newFact, oldFact) {
  const neg = /\b(not|never|no longer|dislike|hate|avoid|don't|doesn't)\b/i;
  const pos = /\b(like|love|prefer|always|want|use|using)\b/i;
  const overlap = similarity(newFact, oldFact);
  if (overlap < 0.35) return false;
  return (neg.test(newFact) && pos.test(oldFact)) || (pos.test(newFact) && neg.test(oldFact));
}

function extractConciseFacts(messages, { minFactLength = 15, maxFactLength = 280 } = {}) {
  const facts = [];
  for (const msg of messages) {
    if (msg.role !== "user" && msg.role !== "assistant") continue;
    const text = stripMemoryTags(
      typeof msg.content === "string"
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.filter((b) => b.type === "text").map((b) => b.text).join("\n")
          : ""
    );
    if (!text) continue;
    const sentences = splitSentences(text);
    for (const sentence of sentences) {
      if (sentence.length < minFactLength || sentence.length > maxFactLength) continue;
      if (!shouldCapture(sentence)) continue;
      if (!facts.some((f) => similarity(f, sentence) > 0.9)) facts.push(sentence);
    }
  }
  return facts;
}

// ---------------------------------------------------------------------------
// LLM prompts & functions
// ---------------------------------------------------------------------------

const FACT_EXTRACTION_PROMPT = `You are a Developer Knowledge Organizer. Your task is to extract concise, standalone facts from the provided conversation.

Focus on extracting:
1. Developer preferences (tools, languages, frameworks, coding style)
2. Technical decisions (architecture choices, library selections, trade-offs)
3. Project details (stack, structure, deployment, environments)
4. Workflow conventions (branching strategy, CI/CD, review process)
5. Important context (deadlines, constraints, team structure)
6. Personal details (name, role, location, timezone)

Rules:
- Each fact must be a single, self-contained sentence
- Use third person ("The user prefers..." not "I prefer...")
- Be concise — under 100 characters per fact when possible
- Do not extract trivial, obvious, or overly generic statements
- Do not extract code snippets or implementation details
- If no meaningful facts can be extracted, return an empty array

Return a JSON object: { "facts": ["fact1", "fact2", ...] }`;

const MEMORY_UPDATE_PROMPT = `You are a memory manager. You will be given a list of new facts and, for each fact, a set of existing memories that are semantically similar.

For each new fact, decide ONE action:
- ADD: The fact is genuinely new information not covered by any existing memory.
- UPDATE: The fact refines, corrects, or adds detail to an existing memory. Provide the updated text and the ID of the memory to update.
- DELETE: The fact directly contradicts an existing memory (the old memory is now wrong). Provide the ID to delete. The new fact will be stored as a replacement.
- NONE: The fact is already captured by an existing memory — no action needed.

Rules:
- Prefer UPDATE over ADD when an existing memory covers the same topic
- Prefer DELETE over UPDATE when the new fact contradicts the old memory
- Only use NONE when the existing memory already says essentially the same thing
- When updating, merge information — don't lose details from the old memory unless contradicted

Return a JSON object: { "memory": [{ "id": "<existing_memory_id or 'new'>", "text": "<fact text>", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "<text of old memory if UPDATE or DELETE, else null>" }] }`;

function createOpenAIClient(config) {
  const apiKey = resolveEnvVars(config.openaiApiKey);
  if (!apiKey) {
    throw new Error("openaiApiKey is required when captureMode is 'llm'");
  }
  return new OpenAI({ apiKey });
}

async function llmExtractFacts(openai, model, messages) {
  const conversationText = messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => {
      const text =
        typeof m.content === "string"
          ? m.content
          : Array.isArray(m.content)
            ? m.content.filter((b) => b.type === "text").map((b) => b.text).join("\n")
            : "";
      return `${m.role}: ${stripMemoryTags(text)}`;
    })
    .filter((line) => line.length > 6)
    .join("\n");

  if (!conversationText.trim()) return [];

  const response = await openai.chat.completions.create({
    model,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: FACT_EXTRACTION_PROMPT },
      { role: "user", content: conversationText },
    ],
  });

  const raw = response.choices?.[0]?.message?.content;
  if (!raw) return [];

  const parsed = JSON.parse(raw);
  return Array.isArray(parsed.facts) ? parsed.facts.filter((f) => typeof f === "string" && f.trim()) : [];
}

async function llmReconcileMemories(openai, model, facts, db, topK = 3, threshold = 0.3) {
  const factsWithContext = [];

  for (const fact of facts) {
    const nearby = await db.search(fact, topK, threshold);
    const existingMemories = nearby.map((hit) => ({
      id: hit._id,
      text: extractHitContent(hit),
      score: hit._score,
    }));
    factsWithContext.push({ fact, existingMemories });
  }

  const prompt = factsWithContext
    .map((fc, i) => {
      const memLines =
        fc.existingMemories.length > 0
          ? fc.existingMemories.map((m) => `  - [${m.id}] (score: ${m.score.toFixed(2)}): ${m.text}`).join("\n")
          : "  (no existing memories found)";
      return `Fact ${i + 1}: ${fc.fact}\nExisting memories:\n${memLines}`;
    })
    .join("\n\n");

  const response = await openai.chat.completions.create({
    model,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: MEMORY_UPDATE_PROMPT },
      { role: "user", content: prompt },
    ],
  });

  const raw = response.choices?.[0]?.message?.content;
  if (!raw) return [];

  const parsed = JSON.parse(raw);
  return Array.isArray(parsed.memory) ? parsed.memory : [];
}

async function applyMemoryDecisions(decisions, db, logger) {
  const stats = { added: 0, updated: 0, deleted: 0, none: 0 };

  for (const decision of decisions) {
    const event = (decision.event ?? "").toUpperCase();
    const text = decision.text?.trim();

    try {
      switch (event) {
        case "ADD": {
          if (!text) break;
          const id = randomUUID();
          await db.store(id, text, {
            category: detectCategory(text),
            role: "llm-extract",
            capturedAt: new Date().toISOString(),
          });
          stats.added += 1;
          break;
        }
        case "UPDATE": {
          if (!text || !decision.id || decision.id === "new") break;
          await db.update(decision.id, text, {
            category: detectCategory(text),
            role: "llm-extract",
            capturedAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          });
          stats.updated += 1;
          break;
        }
        case "DELETE": {
          if (!decision.id || decision.id === "new") break;
          await db.delete(decision.id);
          if (text) {
            const id = randomUUID();
            await db.store(id, text, {
              category: detectCategory(text),
              role: "llm-extract",
              capturedAt: new Date().toISOString(),
            });
          }
          stats.deleted += 1;
          break;
        }
        case "NONE":
          stats.none += 1;
          break;
        default:
          logger.warn(`pinecone-memory: unknown decision event "${event}"`);
      }
    } catch (err) {
      logger.warn(`pinecone-memory: failed to apply ${event} decision: ${err.message}`);
    }
  }

  return stats;
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
    await this._index.upsertRecords({
      records: [
        {
          _id: id,
          content: text,
          ...metadata,
        },
      ],
    });
  }

  async update(id, text, metadata = {}) {
    await this.store(id, text, metadata);
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
export {
  resolveEnvVars,
  shouldCapture,
  detectCategory,
  stripMemoryTags,
  extractHitContent,
  extractConciseFacts,
  similarity,
  isContradiction,
  PineconeMemoryDB,
  FACT_EXTRACTION_PROMPT,
  MEMORY_UPDATE_PROMPT,
  createOpenAIClient,
  llmExtractFacts,
  llmReconcileMemories,
  applyMemoryDecisions,
};

// ---------------------------------------------------------------------------
// Plugin export (OpenClaw plugin API)
// ---------------------------------------------------------------------------

export default function register(api) {
  const config = api.pluginConfig ?? {};
  const db = new PineconeMemoryDB(config);

  const autoCapture = config.autoCapture !== false;
  const autoRecall = config.autoRecall !== false;
  const topK = config.topK ?? 5;
  const similarityThreshold = config.similarityThreshold ?? 0.3;
  const captureMode = config.captureMode ?? "heuristic";
  const updateThreshold = config.updateThreshold ?? 0.72;
  const deleteThreshold = config.deleteThreshold ?? 0.45;
  const summaryTopK = config.summaryTopK ?? 3;
  const minFactLength = config.minFactLength ?? 15;
  const maxFactLength = config.maxFactLength ?? 280;
  const llmModel = config.llmModel ?? "gpt-4o-mini";

  let openaiClient = null;
  if (captureMode === "llm") {
    try {
      openaiClient = createOpenAIClient(config);
    } catch (err) {
      api.logger.warn(`pinecone-memory: failed to create OpenAI client: ${err.message}; falling back to heuristic`);
    }
  }

  api.logger.info(
    `pinecone-memory: registered (index: ${db.indexName}, ns: ${db.namespace}, autoRecall: ${autoRecall}, autoCapture: ${autoCapture}, captureMode: ${captureMode}${captureMode === "llm" ? `, model: ${llmModel}` : ""})`
  );

  // -------------------------------------------------------------------------
  // Hook: before_agent_start — Recall
  // -------------------------------------------------------------------------
  if (autoRecall) {
    api.on("before_agent_start", async (event) => {
      try {
        const prompt = event.prompt?.trim();
        if (!prompt || prompt.length < 5) return;

        const hits = await db.search(prompt, topK, similarityThreshold);
        if (hits.length === 0) return;

        const lines = hits
          .map((hit) => {
            const content = extractHitContent(hit);
            if (!content) return null;
            const cat = hit.category ? ` [${hit.category}]` : "";
            const score = typeof hit._score === "number" ? hit._score.toFixed(2) : "0.00";
            return `- (${score}${cat}) ${content}`;
          })
          .filter(Boolean);

        if (lines.length === 0) return;

        const block = ["<relevant-memories>", ...lines, "</relevant-memories>"].join("\n");

        api.logger.info(`pinecone-memory: recalled ${lines.length} memor${lines.length === 1 ? "y" : "ies"}`);
        return { prependContext: block };
      } catch (err) {
        api.logger.warn(`pinecone-memory: recall failed: ${err.message}`);
      }
    });
  }

  // -------------------------------------------------------------------------
  // Hook: agent_end — Capture
  // -------------------------------------------------------------------------

  async function heuristicCapture(messages) {
    const facts = extractConciseFacts(messages, { minFactLength, maxFactLength });
    if (facts.length === 0) return;

    let added = 0;
    let updated = 0;
    let deleted = 0;
    let none = 0;

    for (const fact of facts) {
      const category = detectCategory(fact);
      const nearby = await db.search(fact, summaryTopK, deleteThreshold);

      const exactDup = nearby.find(
        (hit) =>
          (typeof hit._score === "number" && hit._score >= db.deduplicationThreshold) ||
          similarity(fact, extractHitContent(hit)) >= 0.95
      );
      if (exactDup) {
        none += 1;
        continue;
      }

      const contradiction = nearby.find((hit) => isContradiction(fact, extractHitContent(hit)));
      if (contradiction && contradiction._score >= deleteThreshold) {
        await db.delete(contradiction._id);
        deleted += 1;
        continue;
      }

      const updateTarget = nearby.find(
        (hit) =>
          (typeof hit._score === "number" && hit._score >= updateThreshold) ||
          similarity(fact, extractHitContent(hit)) >= updateThreshold
      );
      if (updateTarget) {
        await db.update(updateTarget._id, fact, {
          category,
          role: "summary",
          capturedAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        });
        updated += 1;
        continue;
      }

      const id = randomUUID();
      await db.store(id, fact, {
        category,
        role: "summary",
        capturedAt: new Date().toISOString(),
      });
      added += 1;
    }

    api.logger.info(
      `pinecone-memory: capture summary facts=${facts.length} added=${added} updated=${updated} deleted=${deleted} none=${none}`
    );
  }

  if (autoCapture) {
    api.on("agent_end", async (event) => {
      try {
        const messages = (event.messages ?? []).slice(-10);

        if (captureMode === "llm" && openaiClient) {
          try {
            const facts = await llmExtractFacts(openaiClient, llmModel, messages);
            if (facts.length === 0) return;
            const decisions = await llmReconcileMemories(openaiClient, llmModel, facts, db, summaryTopK, similarityThreshold);
            const stats = await applyMemoryDecisions(decisions, db, api.logger);
            api.logger.info(
              `pinecone-memory: llm capture facts=${facts.length} added=${stats.added} updated=${stats.updated} deleted=${stats.deleted} none=${stats.none}`
            );
            return;
          } catch (err) {
            api.logger.warn(`pinecone-memory: llm capture failed, falling back to heuristic: ${err.message}`);
          }
        }

        await heuristicCapture(messages);
      } catch (err) {
        api.logger.warn(`pinecone-memory: capture failed: ${err.message}`);
      }
    });
  }

  // -------------------------------------------------------------------------
  // Tool: memory_store
  // -------------------------------------------------------------------------
  api.registerTool({
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
    async execute(_toolCallId, { text, category }) {
      try {
        await db.ensureIndex();

        const clean = normalizeFact(stripMemoryTags(text));
        if (!clean) {
          return { content: [{ type: "text", text: "Memory is empty after cleanup; nothing stored." }] };
        }

        const dup = await db.isDuplicate(clean);
        if (dup) {
          return {
            content: [{ type: "text", text: `Duplicate detected — a very similar memory already exists (score: ${dup._score.toFixed(2)}, id: ${dup._id}). Not stored.` }],
          };
        }

        const id = randomUUID();
        const cat = category ?? detectCategory(clean);

        await db.store(id, clean, {
          category: cat,
          role: "tool",
          capturedAt: new Date().toISOString(),
        });

        return { content: [{ type: "text", text: `Memory stored (id: ${id}, category: ${cat}).` }] };
      } catch (err) {
        return { content: [{ type: "text", text: `Failed to store memory: ${err.message}` }] };
      }
    },
  });

  // -------------------------------------------------------------------------
  // Tool: memory_search
  // -------------------------------------------------------------------------
  api.registerTool({
    name: "memory_search",
    description: "Search long-term memory for relevant facts, preferences, or decisions.",
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
    async execute(_toolCallId, { query, limit }) {
      try {
        const k = limit ?? topK;
        const hits = await db.search(query, k, similarityThreshold);

        if (hits.length === 0) {
          return { content: [{ type: "text", text: "No matching memories found." }] };
        }

        const formatted = hits
          .map((hit) => ({
            id: hit._id,
            score: (hit._score ?? 0).toFixed(2),
            category: hit.category ?? "unknown",
            content: extractHitContent(hit),
            capturedAt: hit.capturedAt ?? null,
          }))
          .filter((r) => r.content);

        if (formatted.length === 0) {
          return { content: [{ type: "text", text: "No matching memories found." }] };
        }

        return { content: [{ type: "text", text: JSON.stringify(formatted, null, 2) }] };
      } catch (err) {
        return { content: [{ type: "text", text: `Search failed: ${err.message}` }] };
      }
    },
  });

  // -------------------------------------------------------------------------
  // Tool: memory_forget
  // -------------------------------------------------------------------------
  api.registerTool({
    name: "memory_forget",
    description: "Delete a memory by ID, or search for and delete a matching memory.",
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
    async execute(_toolCallId, { memoryId, query }) {
      try {
        await db.ensureIndex();

        if (memoryId) {
          await db.delete(memoryId);
          return { content: [{ type: "text", text: `Memory ${memoryId} deleted.` }] };
        }

        if (!query) {
          return { content: [{ type: "text", text: "Provide either memoryId or query." }] };
        }

        const hits = await db.search(query, 5, similarityThreshold);
        if (hits.length === 0) {
          return { content: [{ type: "text", text: "No matching memories found to delete." }] };
        }

        if (hits.length === 1 || hits[0]._score >= 0.9) {
          const target = hits[0];
          await db.delete(target._id);
          return {
            content: [{ type: "text", text: `Deleted memory (id: ${target._id}, score: ${target._score.toFixed(2)}): "${extractHitContent(target).slice(0, 100)}"` }],
          };
        }

        const candidates = hits.map((hit) => ({
          id: hit._id,
          score: hit._score.toFixed(2),
          content: extractHitContent(hit).slice(0, 100),
        }));

        return {
          content: [{ type: "text", text: `Multiple matches found. Specify a memoryId to delete:\n${JSON.stringify(candidates, null, 2)}` }],
        };
      } catch (err) {
        return { content: [{ type: "text", text: `Forget failed: ${err.message}` }] };
      }
    },
  });

  // -------------------------------------------------------------------------
  // CLI: pinecone-memory search <query> [--limit N]
  // -------------------------------------------------------------------------
  api.registerCli(({ program }) => {
    const cmd = program.command("pinecone-memory").description("Pinecone memory plugin commands");

    cmd
      .command("search <query>")
      .description("Search memories")
      .option("--limit <n>", "Max results", parseInt)
      .action(async (query, opts) => {
        try {
          const limit = opts.limit ?? topK;
          const hits = await db.search(query, limit, similarityThreshold);

          if (hits.length === 0) {
            console.log("No matching memories found.");
            return;
          }

          for (const hit of hits) {
            const content = extractHitContent(hit);
            if (!content) continue;
            const cat = hit.category ? ` [${hit.category}]` : "";
            console.log(`  ${hit._score.toFixed(2)}${cat}  ${hit._id}`);
            console.log(`    ${content}`);
            console.log();
          }
        } catch (err) {
          console.error("Search error:", err.message);
        }
      });

    cmd
      .command("stats")
      .description("Show memory plugin status and configuration")
      .action(async () => {
        try {
          await db.ensureIndex();

          console.log("[pinecone-memory] Configuration:");
          console.log(`  Index:         ${db.indexName}`);
          console.log(`  Namespace:     ${db.namespace}`);
          console.log(`  Auto-capture:  ${autoCapture}`);
          console.log(`  Auto-recall:   ${autoRecall}`);
          console.log(`  Capture mode:  ${captureMode}`);
          if (captureMode === "llm") {
            console.log(`  LLM model:     ${llmModel}`);
          }
          console.log(`  Top-K:         ${topK}`);
          console.log(`  Threshold:     ${similarityThreshold}`);
          console.log(`  Dedup:         ${db.deduplicationThreshold}`);
        } catch (err) {
          console.error("Stats error:", err.message);
        }
      });
  }, { commands: ["pinecone-memory"] });
}
