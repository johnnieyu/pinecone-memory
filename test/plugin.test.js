import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Mock Pinecone SDK
// ---------------------------------------------------------------------------
const mockUpsertRecords = vi.fn().mockResolvedValue({});
const mockSearchRecords = vi.fn().mockResolvedValue({ result: { hits: [] } });
const mockDeleteOne = vi.fn().mockResolvedValue({});

const mockNamespace = vi.fn(() => ({
  upsertRecords: mockUpsertRecords,
  searchRecords: mockSearchRecords,
  deleteOne: mockDeleteOne,
}));

const mockIndex = vi.fn(() => ({ namespace: mockNamespace }));
const mockListIndexes = vi.fn().mockResolvedValue({
  indexes: [{ name: "openclaw-memory" }],
});

vi.mock("@pinecone-database/pinecone", () => ({
  Pinecone: vi.fn().mockImplementation(() => ({
    listIndexes: mockListIndexes,
    index: mockIndex,
  })),
}));

// Stable UUID for assertions
vi.mock("node:crypto", () => ({
  randomUUID: vi.fn(() => "test-uuid-1234"),
}));

// ---------------------------------------------------------------------------
// Mock OpenAI SDK
// ---------------------------------------------------------------------------
const mockChatCreate = vi.fn();

vi.mock("openai", () => ({
  default: vi.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: mockChatCreate,
      },
    },
  })),
}));

const { default: register } = await import("../index.js");

// ---------------------------------------------------------------------------
// Helpers — mock OpenClaw plugin API
// ---------------------------------------------------------------------------
function createMockApi(configOverrides = {}) {
  const hooks = {};
  const tools = {};
  let cliSetup = null;

  return {
    pluginConfig: { pineconeApiKey: "test-key", ...configOverrides },
    logger: {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    },
    on(name, handler) {
      hooks[name] = handler;
    },
    registerTool(tool) {
      tools[tool.name] = tool;
    },
    registerCli(setup) {
      cliSetup = setup;
    },
    hooks,
    tools,
    cliSetup,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("plugin – register", () => {
  let api;

  beforeEach(() => {
    vi.clearAllMocks();
    mockListIndexes.mockResolvedValue({
      indexes: [{ name: "openclaw-memory" }],
    });
    api = createMockApi();
    register(api);
  });

  it("registers hooks, tools, and logs init message", () => {
    expect(api.hooks.before_agent_start).toBeDefined();
    expect(api.hooks.agent_end).toBeDefined();
    expect(api.tools.memory_store).toBeDefined();
    expect(api.tools.memory_search).toBeDefined();
    expect(api.tools.memory_forget).toBeDefined();
    expect(api.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("pinecone-memory: registered")
    );
  });

  // -------------------------------------------------------------------------
  // Recall hook (before_agent_start)
  // -------------------------------------------------------------------------
  describe("recall hook", () => {
    it("skips short prompts", async () => {
      const result = await api.hooks.before_agent_start({ prompt: "hi" });
      expect(result).toBeUndefined();
      expect(mockSearchRecords).not.toHaveBeenCalled();
    });

    it("returns prependContext with memory tags when hits found", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [
            { _id: "a", _score: 0.85, content: "User prefers dark mode", category: "preference" },
          ],
        },
      });
      const result = await api.hooks.before_agent_start({
        prompt: "What theme do I like?",
      });
      expect(result.prependContext).toContain("<relevant-memories>");
      expect(result.prependContext).toContain("0.85");
      expect(result.prependContext).toContain("[preference]");
      expect(result.prependContext).toContain("User prefers dark mode");
    });

    it("uses fallback content fields and never renders undefined", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [
            { _id: "a", _score: 0.85, fields: { content: "Memory from fields" }, category: "fact" },
            { _id: "b", _score: 0.82, metadata: { content: "Memory from metadata" }, category: "fact" },
            { _id: "c", _score: 0.8 },
          ],
        },
      });
      const result = await api.hooks.before_agent_start({ prompt: "recall" });
      expect(result.prependContext).toContain("Memory from fields");
      expect(result.prependContext).toContain("Memory from metadata");
      expect(result.prependContext).not.toContain("undefined");
    });

    it("returns undefined when no hits", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: { hits: [] },
      });
      const result = await api.hooks.before_agent_start({
        prompt: "Tell me something random",
      });
      expect(result).toBeUndefined();
    });

    it("handles errors gracefully", async () => {
      mockSearchRecords.mockRejectedValueOnce(new Error("network down"));
      const result = await api.hooks.before_agent_start({
        prompt: "Some valid prompt here",
      });
      expect(result).toBeUndefined();
      expect(api.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining("recall failed")
      );
    });
  });

  // -------------------------------------------------------------------------
  // Capture hook (agent_end)
  // -------------------------------------------------------------------------
  describe("capture hook", () => {
    it("extracts text from messages and stores capturable content", async () => {
      mockSearchRecords.mockResolvedValue({ result: { hits: [] } }); // no dup
      await api.hooks.agent_end({
        messages: [
          { role: "user", content: "I always prefer using TypeScript for new projects" },
        ],
      });
      expect(mockUpsertRecords).toHaveBeenCalledWith({
        records: [
          expect.objectContaining({
            _id: "test-uuid-1234",
            content: "I always prefer using TypeScript for new projects",
            category: "preference",
          }),
        ],
      });
    });

    it("strips memory tags before evaluating", async () => {
      mockSearchRecords.mockResolvedValue({ result: { hits: [] } });
      await api.hooks.agent_end({
        messages: [
          {
            role: "user",
            content:
              "<relevant-memories>old stuff</relevant-memories> I always prefer dark mode",
          },
        ],
      });
      // The stored content should not contain the memory tags
      const storedRecord = mockUpsertRecords.mock.calls[0][0].records[0];
      expect(storedRecord.content).not.toContain("<relevant-memories>");
      expect(storedRecord.content).toBe("I always prefer dark mode");
    });

    it("skips messages that do not match capture patterns", async () => {
      await api.hooks.agent_end({
        messages: [{ role: "user", content: "The quick brown fox jumps over the lazy dog" }],
      });
      expect(mockUpsertRecords).not.toHaveBeenCalled();
    });

    it("skips duplicates", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [{ _id: "existing", _score: 0.99, content: "already there" }],
        },
      });
      await api.hooks.agent_end({
        messages: [
          { role: "user", content: "I always prefer using TypeScript for new projects" },
        ],
      });
      expect(mockUpsertRecords).not.toHaveBeenCalled();
    });

    it("handles array content blocks", async () => {
      mockSearchRecords.mockResolvedValue({ result: { hits: [] } });
      await api.hooks.agent_end({
        messages: [
          {
            role: "assistant",
            content: [
              { type: "text", text: "I decided to go with Redis for caching" },
            ],
          },
        ],
      });
      expect(mockUpsertRecords).toHaveBeenCalled();
    });

    it("updates an existing nearby memory instead of adding", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [{ _id: "existing-id", _score: 0.88, content: "I prefer dark mode in editors" }],
        },
      });

      await api.hooks.agent_end({
        messages: [{ role: "user", content: "I always prefer dark mode in my editor" }],
      });

      const record = mockUpsertRecords.mock.calls[0][0].records[0];
      expect(record._id).toBe("existing-id");
      expect(mockDeleteOne).not.toHaveBeenCalled();
    });

    it("deletes contradictory memory", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [{ _id: "existing-id", _score: 0.6, content: "I like dark mode" }],
        },
      });

      await api.hooks.agent_end({
        messages: [{ role: "user", content: "I dislike dark mode now" }],
      });

      expect(mockDeleteOne).toHaveBeenCalledWith("existing-id");
    });
  });

  // -------------------------------------------------------------------------
  // memory_store tool
  // -------------------------------------------------------------------------
  describe("memory_store tool", () => {
    it("stores a new memory", async () => {
      mockSearchRecords.mockResolvedValueOnce({ result: { hits: [] } });
      const result = await api.tools.memory_store.execute("call-1", {
        text: "Use bun instead of npm",
        category: "preference",
      });
      expect(result.content[0].text).toContain("Memory stored");
      expect(result.content[0].text).toContain("test-uuid-1234");
      expect(mockUpsertRecords).toHaveBeenCalled();
    });

    it("rejects duplicates", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [{ _id: "dup-id", _score: 0.97, content: "dup" }],
        },
      });
      const result = await api.tools.memory_store.execute("call-1", {
        text: "Use bun instead of npm",
      });
      expect(result.content[0].text).toContain("Duplicate detected");
      expect(mockUpsertRecords).not.toHaveBeenCalled();
    });

    it("auto-detects category when not provided", async () => {
      mockSearchRecords.mockResolvedValueOnce({ result: { hits: [] } });
      await api.tools.memory_store.execute("call-1", {
        text: "There is a bug in the auth module",
      });
      const storedRecord = mockUpsertRecords.mock.calls[0][0].records[0];
      expect(storedRecord.category).toBe("technical");
    });
  });

  // -------------------------------------------------------------------------
  // memory_search tool
  // -------------------------------------------------------------------------
  describe("memory_search tool", () => {
    it("returns formatted results", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [
            {
              _id: "mem-1",
              _score: 0.8,
              content: "Prefers dark mode",
              category: "preference",
              capturedAt: "2025-01-01T00:00:00Z",
            },
          ],
        },
      });
      const result = await api.tools.memory_search.execute("call-1", { query: "theme" });
      const parsed = JSON.parse(result.content[0].text);
      expect(parsed).toHaveLength(1);
      expect(parsed[0].id).toBe("mem-1");
      expect(parsed[0].score).toBe("0.80");
    });

    it("handles empty results", async () => {
      mockSearchRecords.mockResolvedValueOnce({ result: { hits: [] } });
      const result = await api.tools.memory_search.execute("call-1", { query: "nothing" });
      expect(result.content[0].text).toBe("No matching memories found.");
    });
  });

  // -------------------------------------------------------------------------
  // memory_forget tool
  // -------------------------------------------------------------------------
  describe("memory_forget tool", () => {
    it("deletes by ID", async () => {
      const result = await api.tools.memory_forget.execute("call-1", {
        memoryId: "abc-123",
      });
      expect(mockDeleteOne).toHaveBeenCalledWith("abc-123");
      expect(result.content[0].text).toContain("abc-123 deleted");
    });

    it("auto-deletes high-confidence match from query", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [
            { _id: "auto-del", _score: 0.95, content: "old preference" },
          ],
        },
      });
      const result = await api.tools.memory_forget.execute("call-1", {
        query: "old preference",
      });
      expect(mockDeleteOne).toHaveBeenCalledWith("auto-del");
      expect(result.content[0].text).toContain("Deleted memory");
    });

    it("lists candidates for ambiguous matches", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [
            { _id: "c1", _score: 0.7, content: "candidate 1" },
            { _id: "c2", _score: 0.65, content: "candidate 2" },
            { _id: "c3", _score: 0.6, content: "candidate 3" },
          ],
        },
      });
      const result = await api.tools.memory_forget.execute("call-1", {
        query: "something vague",
      });
      expect(result.content[0].text).toContain("Multiple matches found");
      expect(result.content[0].text).toContain("c1");
      expect(result.content[0].text).toContain("c2");
      expect(mockDeleteOne).not.toHaveBeenCalled();
    });

    it("requires memoryId or query", async () => {
      const result = await api.tools.memory_forget.execute("call-1", {});
      expect(result.content[0].text).toContain("Provide either memoryId or query");
    });

    it("handles no matches for query", async () => {
      mockSearchRecords.mockResolvedValueOnce({ result: { hits: [] } });
      const result = await api.tools.memory_forget.execute("call-1", {
        query: "nonexistent",
      });
      expect(result.content[0].text).toContain("No matching memories found");
    });
  });
});

// ---------------------------------------------------------------------------
// autoCapture / autoRecall disabled
// ---------------------------------------------------------------------------
describe("plugin – disabled hooks", () => {
  it("does not register recall hook when autoRecall is false", () => {
    const api = createMockApi({ autoRecall: false });
    register(api);
    expect(api.hooks.before_agent_start).toBeUndefined();
    expect(api.hooks.agent_end).toBeDefined();
  });

  it("does not register capture hook when autoCapture is false", () => {
    const api = createMockApi({ autoCapture: false });
    register(api);
    expect(api.hooks.before_agent_start).toBeDefined();
    expect(api.hooks.agent_end).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// LLM capture mode
// ---------------------------------------------------------------------------
describe("plugin – LLM capture mode", () => {
  let api;

  beforeEach(() => {
    vi.clearAllMocks();
    mockListIndexes.mockResolvedValue({
      indexes: [{ name: "openclaw-memory" }],
    });
  });

  it("uses LLM extraction when captureMode is llm", async () => {
    // Step 1: extraction returns facts
    mockChatCreate.mockResolvedValueOnce({
      choices: [{ message: { content: JSON.stringify({ facts: ["The user prefers dark mode"] }) } }],
    });
    // Step 2: reconciliation returns ADD decision
    mockSearchRecords.mockResolvedValue({ result: { hits: [] } });
    mockChatCreate.mockResolvedValueOnce({
      choices: [{
        message: {
          content: JSON.stringify({
            memory: [{ id: "new", text: "The user prefers dark mode", event: "ADD", old_memory: null }],
          }),
        },
      }],
    });

    api = createMockApi({ captureMode: "llm", openaiApiKey: "sk-test-key" });
    register(api);

    await api.hooks.agent_end({
      messages: [{ role: "user", content: "I prefer dark mode in all my editors" }],
    });

    expect(mockChatCreate).toHaveBeenCalledTimes(2);
    expect(mockUpsertRecords).toHaveBeenCalledWith({
      records: [
        expect.objectContaining({
          content: "The user prefers dark mode",
          role: "llm-extract",
        }),
      ],
    });
    expect(api.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("llm capture")
    );
  });

  it("falls back to heuristic when LLM extraction fails", async () => {
    mockChatCreate.mockRejectedValueOnce(new Error("API rate limited"));
    mockSearchRecords.mockResolvedValue({ result: { hits: [] } });

    api = createMockApi({ captureMode: "llm", openaiApiKey: "sk-test-key" });
    register(api);

    await api.hooks.agent_end({
      messages: [{ role: "user", content: "I always prefer using TypeScript for new projects" }],
    });

    expect(api.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("llm capture failed, falling back to heuristic")
    );
    // Heuristic should still store the memory
    expect(mockUpsertRecords).toHaveBeenCalled();
  });

  it("falls back to heuristic when openaiApiKey is missing", async () => {
    mockSearchRecords.mockResolvedValue({ result: { hits: [] } });

    api = createMockApi({ captureMode: "llm" });
    register(api);

    expect(api.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("failed to create OpenAI client")
    );

    await api.hooks.agent_end({
      messages: [{ role: "user", content: "I always prefer using TypeScript for new projects" }],
    });

    // Should fall back to heuristic since openaiClient is null
    expect(mockChatCreate).not.toHaveBeenCalled();
    expect(mockUpsertRecords).toHaveBeenCalled();
  });

  it("does not call OpenAI when captureMode is heuristic", async () => {
    mockSearchRecords.mockResolvedValue({ result: { hits: [] } });

    api = createMockApi({ captureMode: "heuristic" });
    register(api);

    await api.hooks.agent_end({
      messages: [{ role: "user", content: "I always prefer using TypeScript for new projects" }],
    });

    expect(mockChatCreate).not.toHaveBeenCalled();
    expect(mockUpsertRecords).toHaveBeenCalled();
  });

  it("skips capture when LLM returns no facts", async () => {
    mockChatCreate.mockResolvedValueOnce({
      choices: [{ message: { content: JSON.stringify({ facts: [] }) } }],
    });

    api = createMockApi({ captureMode: "llm", openaiApiKey: "sk-test-key" });
    register(api);

    await api.hooks.agent_end({
      messages: [{ role: "user", content: "Hello, how are you?" }],
    });

    // Extraction returned empty facts, so no reconciliation or storage
    expect(mockChatCreate).toHaveBeenCalledTimes(1);
    expect(mockUpsertRecords).not.toHaveBeenCalled();
  });

  it("LLM capture handles UPDATE and DELETE decisions", async () => {
    // Extraction
    mockChatCreate.mockResolvedValueOnce({
      choices: [{ message: { content: JSON.stringify({ facts: ["User now prefers Bun", "User dislikes npm"] }) } }],
    });
    // Search returns existing memories
    mockSearchRecords
      .mockResolvedValueOnce({
        result: { hits: [{ _id: "mem-1", _score: 0.85, content: "User prefers npm" }] },
      })
      .mockResolvedValueOnce({
        result: { hits: [{ _id: "mem-2", _score: 0.75, content: "User likes npm" }] },
      });
    // Reconciliation
    mockChatCreate.mockResolvedValueOnce({
      choices: [{
        message: {
          content: JSON.stringify({
            memory: [
              { id: "mem-1", text: "User now prefers Bun", event: "UPDATE", old_memory: "User prefers npm" },
              { id: "mem-2", text: "User dislikes npm", event: "DELETE", old_memory: "User likes npm" },
            ],
          }),
        },
      }],
    });

    api = createMockApi({ captureMode: "llm", openaiApiKey: "sk-test-key" });
    register(api);

    await api.hooks.agent_end({
      messages: [{ role: "user", content: "I now prefer Bun and I dislike npm" }],
    });

    // UPDATE: upsert with existing id
    expect(mockUpsertRecords).toHaveBeenCalledWith({
      records: [expect.objectContaining({ _id: "mem-1", content: "User now prefers Bun" })],
    });
    // DELETE: delete old + store replacement
    expect(mockDeleteOne).toHaveBeenCalledWith("mem-2");
    expect(api.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("llm capture")
    );
  });

  it("includes model in registration log when in LLM mode", () => {
    api = createMockApi({ captureMode: "llm", openaiApiKey: "sk-test-key", llmModel: "gpt-4o" });
    register(api);

    expect(api.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("model: gpt-4o")
    );
  });
});
