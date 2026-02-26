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
const mockListIndexes = vi.fn();

vi.mock("@pinecone-database/pinecone", () => ({
  Pinecone: vi.fn().mockImplementation(() => ({
    listIndexes: mockListIndexes,
    index: mockIndex,
  })),
}));

const { PineconeMemoryDB } = await import("../index.js");

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("PineconeMemoryDB", () => {
  let db;

  beforeEach(() => {
    vi.clearAllMocks();
    mockListIndexes.mockResolvedValue({
      indexes: [{ name: "openclaw-memory" }],
    });
    db = new PineconeMemoryDB({ pineconeApiKey: "test-key" });
    // Reset cached promise so each test starts fresh
    db._ready = null;
  });

  // -------------------------------------------------------------------------
  // ensureIndex
  // -------------------------------------------------------------------------
  describe("ensureIndex", () => {
    it("connects when the index exists", async () => {
      await db.ensureIndex();
      expect(mockListIndexes).toHaveBeenCalled();
      expect(mockIndex).toHaveBeenCalledWith("openclaw-memory");
    });

    it("throws when the index does not exist", async () => {
      mockListIndexes.mockResolvedValueOnce({ indexes: [] });
      await expect(db.ensureIndex()).rejects.toThrow(
        /Index "openclaw-memory" not found/
      );
    });

    it("caches the ready promise on subsequent calls", async () => {
      await db.ensureIndex();
      await db.ensureIndex();
      // listIndexes should only be called once because the promise is cached
      expect(mockListIndexes).toHaveBeenCalledTimes(1);
    });
  });

  // -------------------------------------------------------------------------
  // store
  // -------------------------------------------------------------------------
  describe("store", () => {
    it("calls upsertRecords with the correct shape", async () => {
      await db.store("id-1", "hello world", { category: "fact" });
      expect(mockUpsertRecords).toHaveBeenCalledWith({
        records: [
          { _id: "id-1", content: "hello world", category: "fact" },
        ],
      });
    });
  });

  // -------------------------------------------------------------------------
  // search
  // -------------------------------------------------------------------------
  describe("search", () => {
    it("returns hits above the threshold", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: {
          hits: [
            { _id: "a", _score: 0.8, content: "match" },
            { _id: "b", _score: 0.1, content: "no match" },
          ],
        },
      });
      const results = await db.search("test", 5, 0.3);
      expect(results).toHaveLength(1);
      expect(results[0]._id).toBe("a");
    });

    it("returns empty array when no hits meet threshold", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: { hits: [{ _id: "a", _score: 0.1, content: "low" }] },
      });
      const results = await db.search("test", 5, 0.3);
      expect(results).toHaveLength(0);
    });
  });

  // -------------------------------------------------------------------------
  // delete
  // -------------------------------------------------------------------------
  describe("delete", () => {
    it("calls deleteOne with the id", async () => {
      await db.delete("id-1");
      expect(mockDeleteOne).toHaveBeenCalledWith("id-1");
    });
  });

  // -------------------------------------------------------------------------
  // isDuplicate
  // -------------------------------------------------------------------------
  describe("isDuplicate", () => {
    it("returns the hit when above dedup threshold", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: { hits: [{ _id: "dup", _score: 0.98, content: "dup text" }] },
      });
      const result = await db.isDuplicate("dup text");
      expect(result).toEqual(
        expect.objectContaining({ _id: "dup", _score: 0.98 })
      );
    });

    it("returns null when below dedup threshold", async () => {
      mockSearchRecords.mockResolvedValueOnce({
        result: { hits: [{ _id: "x", _score: 0.5, content: "text" }] },
      });
      const result = await db.isDuplicate("text");
      expect(result).toBeNull();
    });
  });
});
