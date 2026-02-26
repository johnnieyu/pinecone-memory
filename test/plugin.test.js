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
const mockCreateIndexForModel = vi.fn().mockResolvedValue({});

vi.mock("@pinecone-database/pinecone", () => ({
  Pinecone: vi.fn().mockImplementation(() => ({
    listIndexes: mockListIndexes,
    createIndexForModel: mockCreateIndexForModel,
    index: mockIndex,
  })),
}));

// Stable UUID for assertions
vi.mock("node:crypto", () => ({
  randomUUID: vi.fn(() => "test-uuid-1234"),
}));

const { default: activate } = await import("../index.js");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function createMockContext(configOverrides = {}) {
  const hooks = {};
  const tools = {};
  const commands = {};

  return {
    config: { pineconeApiKey: "test-key", ...configOverrides },
    onHook(name, handler) {
      hooks[name] = handler;
    },
    registerTool(tool) {
      tools[tool.name] = tool;
    },
    registerCommand(cmd) {
      commands[cmd.name] = cmd;
    },
    hooks,
    tools,
    commands,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("plugin – activate", () => {
  let ctx;

  beforeEach(() => {
    vi.clearAllMocks();
    mockListIndexes.mockResolvedValue({
      indexes: [{ name: "openclaw-memory" }],
    });
    ctx = createMockContext();
    activate(ctx);
  });

  it("registers hooks, tools, and commands", () => {
    expect(ctx.hooks.before_agent_start).toBeDefined();
    expect(ctx.hooks.agent_end).toBeDefined();
    expect(ctx.tools.memory_store).toBeDefined();
    expect(ctx.tools.memory_search).toBeDefined();
    expect(ctx.tools.memory_forget).toBeDefined();
    expect(ctx.commands["pinecone-memory search"]).toBeDefined();
    expect(ctx.commands["pinecone-memory stats"]).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // Recall hook (before_agent_start)
  // -------------------------------------------------------------------------
  describe("recall hook", () => {
    it("skips short prompts", async () => {
      const result = await ctx.hooks.before_agent_start({ prompt: "hi" });
      expect(result).toEqual({});
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
      const result = await ctx.hooks.before_agent_start({
        prompt: "What theme do I like?",
      });
      expect(result.prependContext).toContain("<relevant-memories>");
      expect(result.prependContext).toContain("0.85");
      expect(result.prependContext).toContain("[preference]");
      expect(result.prependContext).toContain("User prefers dark mode");
    });

    it("returns empty object when no hits", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: { hits: [] },
      });
      const result = await ctx.hooks.before_agent_start({
        prompt: "Tell me something random",
      });
      expect(result).toEqual({});
    });

    it("handles errors gracefully", async () => {
      mockSearchRecords.mockRejectedValueOnce(new Error("network down"));
      const result = await ctx.hooks.before_agent_start({
        prompt: "Some valid prompt here",
      });
      expect(result).toEqual({});
    });
  });

  // -------------------------------------------------------------------------
  // Capture hook (agent_end)
  // -------------------------------------------------------------------------
  describe("capture hook", () => {
    it("extracts text from messages and stores capturable content", async () => {
      mockSearchRecords.mockResolvedValue({ result: { hits: [] } }); // no dup
      await ctx.hooks.agent_end({
        messages: [
          { role: "user", content: "I always prefer using TypeScript for new projects" },
        ],
      });
      expect(mockUpsertRecords).toHaveBeenCalledWith([
        expect.objectContaining({
          _id: "test-uuid-1234",
          content: "I always prefer using TypeScript for new projects",
          category: "preference",
        }),
      ]);
    });

    it("strips memory tags before evaluating", async () => {
      mockSearchRecords.mockResolvedValue({ result: { hits: [] } });
      await ctx.hooks.agent_end({
        messages: [
          {
            role: "user",
            content:
              "<relevant-memories>old stuff</relevant-memories> I always prefer dark mode",
          },
        ],
      });
      // The stored content should not contain the memory tags
      const storedRecord = mockUpsertRecords.mock.calls[0][0][0];
      expect(storedRecord.content).not.toContain("<relevant-memories>");
      expect(storedRecord.content).toBe("I always prefer dark mode");
    });

    it("skips messages that do not match capture patterns", async () => {
      await ctx.hooks.agent_end({
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
      await ctx.hooks.agent_end({
        messages: [
          { role: "user", content: "I always prefer using TypeScript for new projects" },
        ],
      });
      expect(mockUpsertRecords).not.toHaveBeenCalled();
    });

    it("handles array content blocks", async () => {
      mockSearchRecords.mockResolvedValue({ result: { hits: [] } });
      await ctx.hooks.agent_end({
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
  });

  // -------------------------------------------------------------------------
  // memory_store tool
  // -------------------------------------------------------------------------
  describe("memory_store tool", () => {
    it("stores a new memory", async () => {
      mockSearchRecords.mockResolvedValueOnce({ result: { hits: [] } });
      const result = await ctx.tools.memory_store.execute({
        text: "Use bun instead of npm",
        category: "preference",
      });
      expect(result.result).toContain("Memory stored");
      expect(result.result).toContain("test-uuid-1234");
      expect(mockUpsertRecords).toHaveBeenCalled();
    });

    it("rejects duplicates", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [{ _id: "dup-id", _score: 0.97, content: "dup" }],
        },
      });
      const result = await ctx.tools.memory_store.execute({
        text: "Use bun instead of npm",
      });
      expect(result.result).toContain("Duplicate detected");
      expect(mockUpsertRecords).not.toHaveBeenCalled();
    });

    it("auto-detects category when not provided", async () => {
      mockSearchRecords.mockResolvedValueOnce({ result: { hits: [] } });
      await ctx.tools.memory_store.execute({
        text: "There is a bug in the auth module",
      });
      const storedRecord = mockUpsertRecords.mock.calls[0][0][0];
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
      const result = await ctx.tools.memory_search.execute({ query: "theme" });
      const parsed = JSON.parse(result.result);
      expect(parsed).toHaveLength(1);
      expect(parsed[0].id).toBe("mem-1");
      expect(parsed[0].score).toBe("0.80");
    });

    it("handles empty results", async () => {
      mockSearchRecords.mockResolvedValueOnce({ result: { hits: [] } });
      const result = await ctx.tools.memory_search.execute({ query: "nothing" });
      expect(result.result).toBe("No matching memories found.");
    });
  });

  // -------------------------------------------------------------------------
  // memory_forget tool
  // -------------------------------------------------------------------------
  describe("memory_forget tool", () => {
    it("deletes by ID", async () => {
      const result = await ctx.tools.memory_forget.execute({
        memoryId: "abc-123",
      });
      expect(mockDeleteOne).toHaveBeenCalledWith("abc-123");
      expect(result.result).toContain("abc-123 deleted");
    });

    it("auto-deletes high-confidence match from query", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [
            { _id: "auto-del", _score: 0.95, content: "old preference" },
          ],
        },
      });
      const result = await ctx.tools.memory_forget.execute({
        query: "old preference",
      });
      expect(mockDeleteOne).toHaveBeenCalledWith("auto-del");
      expect(result.result).toContain("Deleted memory");
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
      const result = await ctx.tools.memory_forget.execute({
        query: "something vague",
      });
      expect(result.result).toContain("Multiple matches found");
      expect(result.result).toContain("c1");
      expect(result.result).toContain("c2");
      expect(mockDeleteOne).not.toHaveBeenCalled();
    });

    it("requires memoryId or query", async () => {
      const result = await ctx.tools.memory_forget.execute({});
      expect(result.error).toContain("Provide either memoryId or query");
    });

    it("handles no matches for query", async () => {
      mockSearchRecords.mockResolvedValueOnce({ result: { hits: [] } });
      const result = await ctx.tools.memory_forget.execute({
        query: "nonexistent",
      });
      expect(result.result).toContain("No matching memories found");
    });
  });
});

// ---------------------------------------------------------------------------
// autoCapture / autoRecall disabled
// ---------------------------------------------------------------------------
describe("plugin – disabled hooks", () => {
  it("does not register recall hook when autoRecall is false", () => {
    const ctx = createMockContext({ autoRecall: false });
    activate(ctx);
    expect(ctx.hooks.before_agent_start).toBeUndefined();
    expect(ctx.hooks.agent_end).toBeDefined();
  });

  it("does not register capture hook when autoCapture is false", () => {
    const ctx = createMockContext({ autoCapture: false });
    activate(ctx);
    expect(ctx.hooks.before_agent_start).toBeDefined();
    expect(ctx.hooks.agent_end).toBeUndefined();
  });
});
