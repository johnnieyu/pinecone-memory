import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  resolveEnvVars,
  shouldCapture,
  detectCategory,
  stripMemoryTags,
  extractHitContent,
  extractConciseFacts,
  similarity,
  isContradiction,
  llmExtractFacts,
  llmReconcileMemories,
  applyMemoryDecisions,
} from "../index.js";

// ---------------------------------------------------------------------------
// resolveEnvVars
// ---------------------------------------------------------------------------
describe("resolveEnvVars", () => {
  beforeEach(() => {
    process.env.__TEST_VAR = "hello";
  });
  afterEach(() => {
    delete process.env.__TEST_VAR;
  });

  it("substitutes ${VAR} with the env value", () => {
    expect(resolveEnvVars("key=${__TEST_VAR}")).toBe("key=hello");
  });

  it("replaces missing vars with empty string", () => {
    expect(resolveEnvVars("key=${__MISSING_VAR_XYZ}")).toBe("key=");
  });

  it("passes through non-strings unchanged", () => {
    expect(resolveEnvVars(42)).toBe(42);
    expect(resolveEnvVars(null)).toBe(null);
    expect(resolveEnvVars(undefined)).toBe(undefined);
  });

  it("handles multiple substitutions", () => {
    process.env.__TEST_B = "world";
    expect(resolveEnvVars("${__TEST_VAR} ${__TEST_B}")).toBe("hello world");
    delete process.env.__TEST_B;
  });
});

// ---------------------------------------------------------------------------
// shouldCapture
// ---------------------------------------------------------------------------
describe("shouldCapture", () => {
  it("returns false for short text", () => {
    expect(shouldCapture("hi")).toBe(false);
  });

  it("returns false for very long text", () => {
    expect(shouldCapture("a".repeat(2001))).toBe(false);
  });

  it("returns false for empty/null input", () => {
    expect(shouldCapture("")).toBe(false);
    expect(shouldCapture(null)).toBe(false);
    expect(shouldCapture(undefined)).toBe(false);
  });

  it("returns true when a capture pattern matches", () => {
    expect(shouldCapture("I always prefer dark mode in my editors")).toBe(true);
    expect(shouldCapture("We decided to go with PostgreSQL for the database")).toBe(true);
    expect(shouldCapture("The project uses a microservices architecture")).toBe(true);
  });

  it("returns false when no patterns match", () => {
    expect(shouldCapture("The quick brown fox jumps over the lazy dog")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// detectCategory
// ---------------------------------------------------------------------------
describe("detectCategory", () => {
  it("detects preference", () => {
    expect(detectCategory("I always prefer tabs over spaces")).toBe("preference");
  });

  it("detects decision", () => {
    expect(detectCategory("We decided to use React")).toBe("decision");
  });

  it("detects project", () => {
    expect(detectCategory("The project uses TypeScript")).toBe("project");
  });

  it("detects technical", () => {
    expect(detectCategory("There is a bug in the login flow")).toBe("technical");
  });

  it("detects fact", () => {
    expect(detectCategory("The server is running on port 3000")).toBe("fact");
  });

  it("falls back to general", () => {
    expect(detectCategory("lorem ipsum dolor sit amet")).toBe("general");
  });
});

// ---------------------------------------------------------------------------
// stripMemoryTags
// ---------------------------------------------------------------------------
describe("stripMemoryTags", () => {
  it("removes relevant-memories tags and content", () => {
    const input = "Before <relevant-memories>\nstuff\n</relevant-memories> After";
    expect(stripMemoryTags(input)).toBe("Before  After");
  });

  it("handles text with no tags", () => {
    expect(stripMemoryTags("plain text")).toBe("plain text");
  });

  it("removes multiple tag blocks", () => {
    const input = "<relevant-memories>a</relevant-memories> mid <relevant-memories>b</relevant-memories>";
    expect(stripMemoryTags(input)).toBe("mid");
  });
});

// ---------------------------------------------------------------------------
// extractHitContent
// ---------------------------------------------------------------------------
describe("extractHitContent", () => {
  it("supports top-level content", () => {
    expect(extractHitContent({ content: "hello" })).toBe("hello");
  });

  it("supports fields.content and metadata.content", () => {
    expect(extractHitContent({ fields: { content: "field content" } })).toBe("field content");
    expect(extractHitContent({ metadata: { content: "meta content" } })).toBe("meta content");
  });

  it("returns empty string when missing", () => {
    expect(extractHitContent({ _id: "x" })).toBe("");
  });
});

// ---------------------------------------------------------------------------
// summary helpers
// ---------------------------------------------------------------------------
describe("summary helpers", () => {
  it("extracts concise facts from messages", () => {
    const facts = extractConciseFacts([
      { role: "user", content: "I always prefer dark mode. I decided to use bun instead of npm." },
    ]);
    expect(facts.length).toBeGreaterThan(0);
    expect(facts.join(" ")).toContain("prefer dark mode");
  });

  it("calculates similarity and contradiction", () => {
    expect(similarity("I like dark mode", "I like dark mode")).toBeGreaterThan(0.9);
    expect(isContradiction("I dislike dark mode", "I like dark mode")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// llmExtractFacts
// ---------------------------------------------------------------------------
describe("llmExtractFacts", () => {
  function mockOpenAI(responseContent) {
    return {
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue({
            choices: [{ message: { content: JSON.stringify(responseContent) } }],
          }),
        },
      },
    };
  }

  it("extracts facts from messages via LLM", async () => {
    const openai = mockOpenAI({ facts: ["The user prefers dark mode", "The project uses TypeScript"] });
    const facts = await llmExtractFacts(openai, "gpt-5-mini", [
      { role: "user", content: "I prefer dark mode. The project uses TypeScript." },
    ]);
    expect(facts).toEqual(["The user prefers dark mode", "The project uses TypeScript"]);
    expect(openai.chat.completions.create).toHaveBeenCalledOnce();
  });

  it("returns empty array when no user messages", async () => {
    const openai = mockOpenAI({ facts: [] });
    const facts = await llmExtractFacts(openai, "gpt-5-mini", []);
    expect(facts).toEqual([]);
    expect(openai.chat.completions.create).not.toHaveBeenCalled();
  });

  it("ignores assistant messages", async () => {
    const openai = mockOpenAI({ facts: [] });
    const facts = await llmExtractFacts(openai, "gpt-5-mini", [
      { role: "assistant", content: "I recommend using TypeScript for this project" },
    ]);
    expect(facts).toEqual([]);
    expect(openai.chat.completions.create).not.toHaveBeenCalled();
  });

  it("returns empty array when LLM returns no choices", async () => {
    const openai = {
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue({ choices: [] }),
        },
      },
    };
    const facts = await llmExtractFacts(openai, "gpt-5-mini", [
      { role: "user", content: "I prefer dark mode in editors" },
    ]);
    expect(facts).toEqual([]);
  });

  it("filters out non-string facts", async () => {
    const openai = mockOpenAI({ facts: ["valid fact", 42, null, "another fact", ""] });
    const facts = await llmExtractFacts(openai, "gpt-5-mini", [
      { role: "user", content: "Some conversation content here" },
    ]);
    expect(facts).toEqual(["valid fact", "another fact"]);
  });

  it("handles array content blocks in messages", async () => {
    const openai = mockOpenAI({ facts: ["The user likes bun"] });
    const facts = await llmExtractFacts(openai, "gpt-5-mini", [
      {
        role: "user",
        content: [{ type: "text", text: "I like using bun" }],
      },
    ]);
    expect(facts).toEqual(["The user likes bun"]);
  });

  it("strips memory tags from message content", async () => {
    const openai = mockOpenAI({ facts: ["The user prefers vim"] });
    await llmExtractFacts(openai, "gpt-5-mini", [
      { role: "user", content: "<relevant-memories>old</relevant-memories> I prefer vim" },
    ]);
    const callArgs = openai.chat.completions.create.mock.calls[0][0];
    const userContent = callArgs.messages[1].content;
    expect(userContent).not.toContain("<relevant-memories>");
    expect(userContent).toContain("prefer vim");
    // Should not include role prefix since we only send user text
    expect(userContent).not.toContain("user:");
  });
});

// ---------------------------------------------------------------------------
// llmReconcileMemories
// ---------------------------------------------------------------------------
describe("llmReconcileMemories", () => {
  function mockOpenAI(responseContent) {
    return {
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue({
            choices: [{ message: { content: JSON.stringify(responseContent) } }],
          }),
        },
      },
    };
  }

  function mockDB(searchResults = []) {
    return {
      search: vi.fn().mockResolvedValue(searchResults),
    };
  }

  it("maps integer IDs to real UUIDs and back", async () => {
    // LLM receives integer IDs and responds with them
    const decisions = {
      memory: [
        { id: "new", text: "The user prefers dark mode", event: "ADD", old_memory: null },
        { id: "0", text: "The project uses Bun", event: "UPDATE", old_memory: "The project uses npm" },
      ],
    };
    const openai = mockOpenAI(decisions);
    const db = mockDB([{ _id: "mem-real-uuid-1", _score: 0.8, content: "The project uses npm" }]);

    const result = await llmReconcileMemories(openai, "gpt-5-mini", ["User prefers dark mode", "Project uses Bun"], db);

    expect(db.search).toHaveBeenCalledTimes(2);
    expect(openai.chat.completions.create).toHaveBeenCalledOnce();

    // Verify LLM received integer IDs, not UUIDs
    const callArgs = openai.chat.completions.create.mock.calls[0][0];
    expect(callArgs.messages[1].content).toContain("id=0");
    expect(callArgs.messages[1].content).not.toContain("mem-real-uuid-1");

    // Verify output maps back to real UUIDs
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe("new");
    expect(result[0].event).toBe("ADD");
    expect(result[1].id).toBe("mem-real-uuid-1");
    expect(result[1].event).toBe("UPDATE");
  });

  it("returns empty array when LLM returns no content", async () => {
    const openai = {
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue({ choices: [{ message: { content: null } }] }),
        },
      },
    };
    const db = mockDB();
    const result = await llmReconcileMemories(openai, "gpt-5-mini", ["a fact"], db);
    expect(result).toEqual([]);
  });

  it("handles facts with no nearby memories", async () => {
    const decisions = {
      memory: [{ id: "new", text: "Brand new fact", event: "ADD", old_memory: null }],
    };
    const openai = mockOpenAI(decisions);
    const db = mockDB([]);

    const result = await llmReconcileMemories(openai, "gpt-5-mini", ["Brand new fact"], db);
    expect(result).toHaveLength(1);

    const callArgs = openai.chat.completions.create.mock.calls[0][0];
    expect(callArgs.messages[1].content).toContain("(no existing memories found)");
  });
});

// ---------------------------------------------------------------------------
// applyMemoryDecisions
// ---------------------------------------------------------------------------
describe("applyMemoryDecisions", () => {
  function mockDB() {
    return {
      store: vi.fn().mockResolvedValue(undefined),
      update: vi.fn().mockResolvedValue(undefined),
      delete: vi.fn().mockResolvedValue(undefined),
    };
  }

  function mockLogger() {
    return { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() };
  }

  it("applies ADD decisions", async () => {
    const db = mockDB();
    const logger = mockLogger();
    const stats = await applyMemoryDecisions(
      [{ id: "new", text: "The user prefers dark mode", event: "ADD", old_memory: null }],
      db,
      logger
    );
    expect(stats.added).toBe(1);
    expect(db.store).toHaveBeenCalledOnce();
    const storeCall = db.store.mock.calls[0];
    expect(storeCall[1]).toBe("The user prefers dark mode");
    expect(storeCall[2].role).toBe("llm-extract");
  });

  it("applies UPDATE decisions", async () => {
    const db = mockDB();
    const logger = mockLogger();
    const stats = await applyMemoryDecisions(
      [{ id: "mem-1", text: "The user prefers Bun over npm", event: "UPDATE", old_memory: "The user uses npm" }],
      db,
      logger
    );
    expect(stats.updated).toBe(1);
    expect(db.update).toHaveBeenCalledWith("mem-1", "The user prefers Bun over npm", expect.objectContaining({ role: "llm-extract" }));
  });

  it("applies DELETE decisions — deletes old and stores new fact", async () => {
    const db = mockDB();
    const logger = mockLogger();
    const stats = await applyMemoryDecisions(
      [{ id: "mem-2", text: "The user dislikes dark mode", event: "DELETE", old_memory: "The user likes dark mode" }],
      db,
      logger
    );
    expect(stats.deleted).toBe(1);
    expect(db.delete).toHaveBeenCalledWith("mem-2");
    expect(db.store).toHaveBeenCalledOnce();
    expect(db.store.mock.calls[0][1]).toBe("The user dislikes dark mode");
  });

  it("counts NONE decisions", async () => {
    const db = mockDB();
    const logger = mockLogger();
    const stats = await applyMemoryDecisions(
      [{ id: "mem-3", text: "Same fact", event: "NONE", old_memory: null }],
      db,
      logger
    );
    expect(stats.none).toBe(1);
    expect(db.store).not.toHaveBeenCalled();
    expect(db.update).not.toHaveBeenCalled();
    expect(db.delete).not.toHaveBeenCalled();
  });

  it("handles mixed decisions", async () => {
    const db = mockDB();
    const logger = mockLogger();
    const stats = await applyMemoryDecisions(
      [
        { id: "new", text: "New fact", event: "ADD", old_memory: null },
        { id: "mem-1", text: "Updated fact", event: "UPDATE", old_memory: "Old" },
        { id: "mem-2", text: "Replacement", event: "DELETE", old_memory: "Contradicted" },
        { id: "mem-3", text: "Duplicate", event: "NONE", old_memory: null },
      ],
      db,
      logger
    );
    expect(stats).toEqual({ added: 1, updated: 1, deleted: 1, none: 1 });
  });

  it("skips ADD when text is empty", async () => {
    const db = mockDB();
    const logger = mockLogger();
    const stats = await applyMemoryDecisions(
      [{ id: "new", text: "", event: "ADD", old_memory: null }],
      db,
      logger
    );
    expect(stats.added).toBe(0);
    expect(db.store).not.toHaveBeenCalled();
  });

  it("skips UPDATE when id is 'new'", async () => {
    const db = mockDB();
    const logger = mockLogger();
    const stats = await applyMemoryDecisions(
      [{ id: "new", text: "Some text", event: "UPDATE", old_memory: null }],
      db,
      logger
    );
    expect(stats.updated).toBe(0);
    expect(db.update).not.toHaveBeenCalled();
  });

  it("logs warning for unknown event types", async () => {
    const db = mockDB();
    const logger = mockLogger();
    await applyMemoryDecisions(
      [{ id: "x", text: "text", event: "UNKNOWN", old_memory: null }],
      db,
      logger
    );
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("unknown decision event"));
  });

  it("continues processing after individual decision failure", async () => {
    const db = mockDB();
    db.store.mockRejectedValueOnce(new Error("network error"));
    const logger = mockLogger();
    const stats = await applyMemoryDecisions(
      [
        { id: "new", text: "Fails", event: "ADD", old_memory: null },
        { id: "new", text: "Succeeds", event: "ADD", old_memory: null },
      ],
      db,
      logger
    );
    expect(stats.added).toBe(1);
    expect(logger.warn).toHaveBeenCalledWith(expect.stringContaining("failed to apply ADD"));
  });
});
