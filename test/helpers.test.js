import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  resolveEnvVars,
  shouldCapture,
  detectCategory,
  stripMemoryTags,
  extractHitContent,
  extractConciseFacts,
  similarity,
  isContradiction,
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
